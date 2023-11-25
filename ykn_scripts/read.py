
cm=0
pa=0
sb=0
fm=0
IS=0
ps=0
passs=0
ic=0

with open('data/group0/group1_unlabelled_moredata.txt', 'r') as file:
    # 
    for line in file:
        content=line.strip() 
        if 'otherTest' not in content:
            if '/CM/' in content or 'cm' in content:
                cm+=1
            if '/SB/' in content:
                sb+=1
            if '/FM/' in content:
                fm+=1
            if '/IS/' in content or 'is' in content:
                IS+=1
            if '/PS/' in content:
                ps+=1
            if '/IC/' in content:
                ic+=1
    print('cm',cm)
    print('sb',sb)
    print('fm',fm)
    print('is',IS)
    print('ps',ps)
    print('ic',ic)
    