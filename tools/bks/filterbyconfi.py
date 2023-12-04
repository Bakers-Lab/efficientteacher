import json

folder='new'


with open('./pred_json_file/'+folder+'/test.bbox.json', "r") as file:
    data=json.load(file)
    
filtered_data = [item for item in data if item['score'] >= 0.014]
with open('./pred_json_file/'+folder+'/new.bbox.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)