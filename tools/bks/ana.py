from tools.bks.utils.metrics import box_iou
from tools.bks.utils.plots import *
from tools.bks.post_process import EXPORT_TPMBDBDL_FILENAME, EXPORT_TGT_FILENAME

label_colors = Colors()

indicator_names = ['GT', 'pred', '一致数量', '一致率', '漏检数量', '漏检率', '错检数量', '错检率', '过杀数量', '过杀率']


def write_gt_area(gt, gt_paths, input_path, label_names):
    path = input_path + '/gt_area.txt'
    with open(path, 'w') as f:
        for gi, g in enumerate(gt):
            area = g[2] * g[3] * 100
            c = label_names[int(g[-1])]
            p = gt_paths[int(g[-2])]
            f.write(str(gi) + ',' + str(area) + ',' + str(c) + ',' + str(p) + ',' + str(int(g[-2])) + '\n')
        f.close()


def format_float(f):
    return '%.4f' % f


def read_files(model_name: str,
               label_names: list,
               input_path='result_txt',
               merge=True,
               merge_file='split_merged_step2_pred_',
               is_deduplicate=True):
    gt_file = open(os.path.join(input_path, 'gt_enhance_' + model_name + '.txt'), 'r')
    # gt_file = open(os.path.join(input_path, 'gt_process.txt'), 'r')
    # pred_file = open(os.path.join(input_path, 'pred_process.txt'), 'r')
    pred_file = open(os.path.join(input_path, 'TPMBDBDL.txt'), 'r')
    # pred_file = open(os.path.join(input_path, 'pred_enhance_yolo.txt'), 'r')
    paths_file = open(os.path.join(input_path, 'paths_' + model_name + '.txt'), 'r')
    gt_lines, pred_lines, paths_lines = gt_file.readlines(), pred_file.readlines(), paths_file.readlines()
    gt_file.close(), pred_file.close(), paths_file.close()

    gt = np.zeros((len(gt_lines), 6))
    pred = np.zeros((len(pred_lines), 8)) if merge else np.zeros((len(pred_lines), 8))

    paths = []
    for i, line in enumerate(paths_lines):
        paths.append(line.replace('\n', '').replace('/test/', '/test_transfered/'))
        # paths.append(
        #     line.replace('\n', '').replace('/train/', '/train_transfered/'))
        # paths.append(
        # line.replace('\n', '').replace('/val/', '/val_transfered/'))
        # 使用yolo转换过后的图片，解决框变形偏移问题
    for i, line in enumerate(gt_lines):
        elems = line.replace('\n', '').split(',')
        gt[i] = np.array(
            [float(elems[0]),
             float(elems[1]),
             float(elems[2]),
             float(elems[3]),
             int(elems[4]),
             int(elems[5])])
    write_gt_area(gt, paths, input_path, label_names)
    gt[:, :4] = xywh2xyxy(gt[:, :4])
    for i, line in enumerate(pred_lines):
        elems = line.replace('\n', '').split(',')
        # if merge:
        pred[i] = np.array([
            float(elems[0]),
            float(elems[1]),
            float(elems[2]),
            float(elems[3]),
            int(float(elems[5])),
            int(float(elems[6])),
            float(elems[4]),
            int(float(elems[8]))
        ])
        # else:
        #     pred[i] = np.array([float(elems[0]), float(elems[1]), float(elems[2]),
        #                         float(elems[3]), int(elems[5]), int(float(elems[6])),
        #                         float(elems[4]), ])
    pred[:, :4] = xywh2xyxy(pred[:, :4])
    if is_deduplicate:
        pred = deduplicate(pred, len(paths))
    return gt, pred, paths


def deduplicate(pred, img_count):
    pred_new = np.zeros(pred.shape)
    count = 0
    for i in range(img_count):
        # imgid
        p = pred[pred[:, -4] == i]
        try:
            for j in range(int(np.max(p[:, -1])) + 1):
                pj = p[p[:, -1] == j]
                s = set()
                for pj_ in pj:
                    if pj_[-3] not in s:
                        s.add(pj_[-3])
                        pred_new[count, :] = pj_
                        count += 1
        except ValueError:
            p
    # print(count)
    return pred_new[:count, :]


def calc_indicators(model_name: str,
                    label_names: list,
                    vis: bool = True,
                    input_path='result_txt',
                    out_path='result_image',
                    classes: int = 10,
                    merge=True,
                    merge_file='split_mergerd_pred_',
                    vis_shape=(1000, 500),
                    batch_size=64,
                    conf_thres=0.0,
                    iou_thres=0.0,
                    draw_label=True,
                    is_dup=True,
                    others=False,
                    pred_to_label_path='pred_to_label.txt',
                    missing_path='missing.txt'):

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gt, pred, paths = read_files(model_name, label_names, input_path, merge, merge_file, is_dup)

    pred_to_label_file = open(os.path.join(input_path, pred_to_label_path), 'w')
    missing_file = open(os.path.join(input_path, missing_path), 'w')
    count = np.zeros((10, classes))
    image_count = int(gt[-1, 4] + 1)
    batch_count = math.ceil(image_count / batch_size)

    count_boxes = {
        'gt_plot': 0,
        'pred_plot': 0,
        'pred_filtered': 0,
        'missing_plot': 0,
        'overkill_plot': 0,
        'overkill_filtered': 0
    }
    gt[:, :4] = ratio_to_coord(gt[:, :4], vis_shape)
    pred[:, :4] = ratio_to_coord(pred[:, :4], vis_shape)
    over_kill = [[], [], [], [], [], []]
    not_over_kill = [[], [], [], [], [], []]
    img_id = 0
    count_missing = 0
    for b in range(batch_count):
        plot_missing, plot_over_kill = None, None
        plot_not_over_kill = None
        images = None
        for k in range(batch_size):
            i = b * batch_size + k
            if i >= image_count:
                break
            if vis:
                origin_image = cv2.imread(paths[i])
                origin_image = origin_image[:, :, ::-1]
                image = cv2.resize(origin_image, vis_shape)
                if images is None:
                    images = image
                else:
                    images = np.vstack((images, image))
            l = gt[gt[:, 4] == i]
            p = pred[pred[:, 4] == i]
            inds = industrial_indicators(l,
                                         p,
                                         box_iou(torch.Tensor(l[:, :4]), torch.Tensor(p[:, :4])),
                                         i,
                                         vis=vis,
                                         threshold=iou_thres,
                                         conf_thres=conf_thres,
                                         merge=merge,
                                         cls=classes)
            k = 0
            for j, ind in enumerate(inds[:6]):
                if j > 1:
                    count[k, :] += ind
                    k += 2
                else:
                    count[k, :] += ind
                    k += 1
            if inds[6] is not None:
                if plot_missing is None:
                    plot_missing = inds[6]
                else:
                    plot_missing = np.vstack((plot_missing, inds[6]))
            if inds[7] is not None:
                if plot_over_kill is None:
                    plot_over_kill = inds[7]
                else:
                    plot_over_kill = np.vstack((plot_over_kill, inds[7]))
            if inds[8] is not None:
                if plot_not_over_kill is None:
                    plot_not_over_kill = inds[8]
                else:
                    plot_not_over_kill = np.vstack((plot_not_over_kill, inds[8]))
            if inds[9] is not None:
                for ind in inds[9]:
                    pred_to_label_file.write(str(int(ind[0])) + ',' + str(int(ind[1])) + ',' + str(int(ind[2])) + '\n')

        try:
            ## 如果plot_over_kill只有一行，那么它是一个一维数组，i就是标量，无法取i[4]等
            if plot_over_kill is not None:
                if plot_over_kill.ndim == 1:
                    temp = len(plot_over_kill)
                    plot_over_kill.resize(1, temp)
                for i in plot_over_kill:
                    over_kill[0].append((i[4] - i[2]) * (i[5] - i[3]) / 5000)
                    over_kill[2].append(label_names[int(i[1])])
                    over_kill[3].append(paths[int(i[0])])
                    over_kill[4].append(int(i[0]))
                    over_kill[1].append((i[6]) * 100)
                    if merge:
                        over_kill[5].append(i[7])
                    else:
                        over_kill[5].append(0)
            for i in plot_not_over_kill:
                not_over_kill[0].append((i[4] - i[2]) * (i[5] - i[3]) / 5000)
                not_over_kill[2].append(label_names[int(i[1])])
                not_over_kill[3].append(paths[int(i[0])])
                not_over_kill[4].append(int(i[0]))
                not_over_kill[1].append((i[6]) * 100)
                if merge:
                    not_over_kill[5].append(i[7])
                else:
                    not_over_kill[5].append(0)
        except TypeError:
            continue

        if vis:
            plot_missing_image(label_names, plot_missing, paths, vis_shape, 36, out_path, model_name, t='missing')
            plot_missing_image(label_names, plot_over_kill, paths, vis_shape, 36, out_path, model_name)
            # plot_missing[:, 2:] = ratio_to_coord(plot_missing[:, 2:], vis_shape)
            # plot_over_kill[:, 2:6] = ratio_to_coord(plot_over_kill[:, 2:6], vis_shape)
            font_size = 36
            # 4列，分别绘制gt、pred、miss、overkill
            annotator = Annotator(np.hstack((images, images, images, images)),
                                  line_width=5,
                                  font_size=font_size,
                                  pil=True,
                                  example=label_names)
            # 编号字体
            img_id_font = check_pil_font(font='Arial.ttf', size=100)
            for k in range(batch_size):
                i = b * batch_size + k
                # annotator.text((0, k*vis_shape[1]), str(i))
                annotator.draw.text((0, k * vis_shape[1]), str(i), fill=(255, 255, 255), font=img_id_font)
                if i >= image_count:
                    break
                # 绘制gt
                ls = gt[gt[:, 4] == i]
                for l in ls:
                    c = int(l[5])
                    # if c != 7 and c != 8:
                    box = l[:4]
                    box[1] = box[1] + k * vis_shape[1]
                    box[3] = box[3] + k * vis_shape[1]
                    box_text = label_names[c] if draw_label else ''
                    annotator.box_label(box.astype(int).tolist(), box_text, color=colors(c))
                    count_boxes['gt_plot'] += 1
                # 绘制pred
                ps = pred[pred[:, 4] == i]
                box_text = ''
                for pi, p in enumerate(ps):
                    if pi == ps.shape[0] - 1:
                        next_box = [0, 0, 0, 0]
                    else:
                        next_box = ps[pi + 1, :4]
                    c = int(p[5])
                    # if c !=7 and c != 8:
                    box = p[:4]
                    box[0] = box[0] + vis_shape[0]
                    box[1] = box[1] + k * vis_shape[1]
                    box[2] = box[2] + vis_shape[0]
                    box[3] = box[3] + k * vis_shape[1]
                    if p[6] >= conf_thres:
                        if next_box[0] == p[0] and next_box[1] == p[1] and next_box[2] == p[2] and next_box[3] == p[3]:
                            if box_text == '':
                                box_text = label_names[c].split('_')[1]
                            else:
                                box_text = box_text + ',' + label_names[c].split('_')[1]
                        else:
                            if box_text == '':
                                box_text = label_names[c].split('_')[1]
                            else:
                                box_text = box_text + ',' + label_names[c].split('_')[1]
                                annotator.box_label(box.astype(int).tolist(),
                                                    box_text if draw_label else '',
                                                    color=colors(c))
                                box_text = ''
                        count_boxes['pred_plot'] += 1
                    else:
                        count_boxes['pred_filtered'] += 1
                # 绘制miss
                if plot_missing is not None:
                    if len(plot_missing.shape) == 1:
                        plot_missing = np.expand_dims(plot_missing, 0)
                    ms = plot_missing[plot_missing[:, 0] == i]
                    for m in ms:
                        c = int(m[1])
                        # if c!=7 and c!=8:
                        box = m[2:6]
                        try:
                            box[0] = box[0] + vis_shape[0] * 2
                            box[1] = box[1] + k * vis_shape[1]
                            box[2] = box[2] + vis_shape[0] * 2
                            box[3] = box[3] + k * vis_shape[1]
                        except IndexError:
                            print(box)
                        box_text = label_names[c] if draw_label else ''
                        try:
                            annotator.box_label(box.astype(int).tolist(), box_text, color=colors(c))
                        except ValueError:
                            print(box)
                        count_boxes['missing_plot'] += 1
                # 绘制overkill
                if plot_over_kill is not None:
                    if len(plot_over_kill.shape) == 1:
                        plot_over_kill = np.expand_dims(plot_over_kill, 0)
                    oks = plot_over_kill[plot_over_kill[:, 0] == i]
                    for o in oks:
                        c = int(o[1])
                        # if c != 7 and c!=8:
                        box = o[2:6]
                        box[0] = box[0] + vis_shape[0] * 3
                        box[1] = box[1] + k * vis_shape[1]
                        box[2] = box[2] + vis_shape[0] * 3
                        box[3] = box[3] + k * vis_shape[1]
                        if o[6] >= conf_thres:
                            if merge:
                                box_text = label_names[c] + ': ' + str(o[6]) if draw_label else str(int(o[7]))
                            else:
                                box_text = label_names[c] if draw_label else str(0)
                            annotator.box_label(box.astype(int).tolist(), box_text, color=colors(c))
                            count_boxes['overkill_plot'] += 1
                        else:
                            count_boxes['overkill_filtered'] += 1
            annotator.im.save(
                os.path.join(out_path, model_name + '_batch' + str(b) + '_conf' + str(conf_thres) + '.jpg'))

        count_missing = write_missing(label_names, missing_file, plot_missing, paths, count_missing)
    # print('框框计数：', count_boxes)
    # print(over_kill_area)
    # print(over_kill_conf)
    # print(not_over_kill_area)
    # print(not_over_kill_conf)
    pred_to_label_file.close()
    write_area_to_file(input_path + '/ok_area.txt', over_kill, mode='w')
    write_area_to_file(input_path + '/nok_area.txt', not_over_kill, mode='w')
    over_kill.insert(0, [1 for _ in range(len(over_kill[0]))])
    not_over_kill.insert(0, [0 for _ in range(len(not_over_kill[0]))])
    write_area_to_file(input_path + '/area.txt', over_kill, mode='w')
    write_area_to_file(input_path + '/area.txt', not_over_kill, mode='a')

    if others:
        overall = np.sum(count, axis=1)
    else:
        overall = np.sum(count[:, :-1], axis=1)
    count = np.hstack((count, np.expand_dims(overall, 1)))
    for j in [3, 5, 7]:
        count[j, :] = count[j - 1, :] / count[0, :]
    count[9, :] = count[8, :] / count[1, :]
    pd.DataFrame(count.T).to_csv(os.path.join(input_path, 'result_' + model_name + '.csv'),
                                 header=indicator_names,
                                 encoding='gb2312')
    print(count.T[-1, :])


def write_missing(label_names, missing_file, missing, paths, count_missing):
    if missing is None:
        return count_missing
    elif len(missing.shape) == 1:
        missing = np.expand_dims(missing, 0)
    for m in missing:
        area = (m[4] - m[2]) / 1000 * (m[5] - m[3]) / 500 * 100
        img_file = paths[int(m[0])].split('/')[-1]
        missing_file.write(
            str(count_missing) + ',' + format_float(area) + ',' + label_names[int(m[1])] + ',' + img_file + ',' +
            str(int(m[0])) + ',' + str(int(m[6])) + '\n')
        count_missing += 1
    return count_missing


def plot_missing_image(label_names, plot_missing, paths, shape, font_size, out_path, model_name, t='overkill'):
    if plot_missing is None:
        return
    if len(plot_missing.shape) == 1:
        plot_missing = np.expand_dims(plot_missing, 0)
    path = os.path.join(out_path, model_name, t)
    if not os.path.exists(path):
        os.makedirs(path)
    for m in plot_missing:
        i = m[0]
        img = cv2.imread(paths[int(i)])
        img_name = paths[int(i)].split('/')[-1]
        img = cv2.resize(img, shape)
        anno = Annotator(img, line_width=5, font_size=font_size, pil=True, example=label_names)
        c = int(m[1])
        box_text = label_names[c]
        anno.box_label(m[2:6].astype(int).tolist(), box_text, color=colors(c))
        anno.im.save(os.path.join(path, img_name))


def write_area_to_file(file, area, mode='w'):
    with open(file, mode) as f:
        for i in range(len(area[0])):
            for a in area:
                f.write(str(a[i]))
                f.write(',')
            f.write('\n')


def ratio_to_coord(boxs, shape):
    for i in range(4):
        boxs[:, i] = boxs[:, i] * shape[i % 2]
    return boxs


def industrial_indicators(labels, pred, iou, image_id, vis=False, threshold=0.0, conf_thres=0.0, merge=True, cls=10):
    n_GT, n_pre, n_cons, n_miss, n_error, n_overkill = \
        np.zeros(cls), np.zeros(cls), np.zeros(cls), np.zeros(cls), np.zeros(cls), np.zeros(cls)
    not_overkill_set = [set() for i in range(cls)]
    is_pred = False
    pred = pred[pred[:, 4] >= conf_thres]
    plot_missing = None
    plot_over_kill = None
    plot_not_over_kill = None
    pred_to_label = []
    for i in range(labels.shape[0]):
        l = labels[i]
        lc, lx1, ly1, lx2, ly2 = int(l[5]), l[0], l[1], l[2], l[3]
        n_GT[lc] += 1
        checked = False
        error = False
        for j in range(pred.shape[0]):
            p = pred[j]
            pc, px1, py1, px2, py2 = int(p[5]), p[0], p[1], p[2], p[3]
            print(f'wzj-debug-pc:{pc}\t{p}')
            if not is_pred:
                n_pre[pc] += 1
            if iou[i, j] > threshold:
                not_overkill_set[pc].add(j)
                if [image_id, p[-1], i] not in pred_to_label:
                    pred_to_label.append([image_id, p[-1], i])
                if lc == pc:
                    if not checked:
                        checked = True
                else:
                    if not error:
                        error = True
        is_pred = True
        if checked:
            n_cons[lc] += 1
        elif error:
            n_error[lc] += 1
        else:
            n_miss[lc] += 1
            m = np.array([image_id, lc, l[0], l[1], l[2], l[3], i])
            if plot_missing is None:
                plot_missing = m
            else:
                plot_missing = np.vstack((plot_missing, m))
    total_set = set()
    for s in not_overkill_set:
        total_set = total_set.union(s)
    for i in range(pred.shape[0]):
        p = pred[i]
        if merge:
            over_kill = np.array([image_id, p[5], p[0], p[1], p[2], p[3], p[6], p[7]])
        else:
            over_kill = np.array([image_id, p[5], p[0], p[1], p[2], p[3], p[6]])
        if i not in total_set:
            if plot_over_kill is None:
                plot_over_kill = over_kill
            else:
                plot_over_kill = np.vstack((plot_over_kill, over_kill))
        else:
            if plot_not_over_kill is None:
                plot_not_over_kill = over_kill
            else:
                plot_not_over_kill = np.vstack((plot_not_over_kill, over_kill))

    for i, s in enumerate(not_overkill_set):
        n_overkill[i] += n_pre[i] - len(s)
    return n_GT, n_pre, n_cons, n_miss, n_error, n_overkill, plot_missing, plot_over_kill, plot_not_over_kill, pred_to_label


def predline_to_str(pred):
    s = ''
    if len(pred.shape) == 1:
        for p in pred[:4]:
            s += format_float(p) + ','
        s += format_float(pred[6] * 100) + ','
        s += str(int(pred[4])) + ','
        s += str(int(pred[5])) + ','
        s += format_float(pred[2] * pred[3] * 100) + ','
        s += str(int(pred[7])) + '\n'
    return s


import yaml


def read_labels(input_path):
    file = open(input_path + 'DATA.yaml')
    yaml_dic = yaml.safe_load(file)
    return yaml_dic['names']


if __name__ == '__main__':
    input_path = 'work_dirs/yolov8_s_syncbn_fast_8xb16-500e_zhck_v2/export-01/'
    out_path = 'work_dirs/yolov8_s_syncbn_fast_8xb16-500e_zhck_v2/export-01/4cols'
    label_names = read_labels(input_path)
    class_count = len(label_names)
    calc_indicators(
        model_name='yolo',
        label_names=label_names,
        vis=True,  # 关键
        conf_thres=0,
        merge=True,  # 关键
        merge_file='split_merged_step3_pred_',
        input_path=input_path,
        out_path=out_path,  # 关键
        draw_label=True,
        classes=class_count,
        is_dup=True,
        others=False)
