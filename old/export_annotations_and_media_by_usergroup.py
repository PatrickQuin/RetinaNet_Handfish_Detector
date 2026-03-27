import re
import requests
import os
import getpass
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import random
# configure some parameters
HOST = "https://squidle.org"    # the SQ+ host/instance you're pointing to
MEDIA_COLLECTION_ID = 13053#13805#13053      # ID of media_collection you want to export
ANNOTATION_SET_ID = 16016
# Get API token from user input for access permission (do not hardcode, save or share this)
try:
    API_TOKEN = "6b21c7427f827202f5c4840169211f2bffe21429024fca871a965bdd" # getpass.getpass(prompt='Enter API Token: ') #
except Exception as error:
    print('ERROR', error)
else:
    print('API Token Entered: ', API_TOKEN)

# API endpoint URL to extract data
url = f'{HOST}/api/media_collection/{MEDIA_COLLECTION_ID}/export?include_columns=["path_best"]'
annotation_url = f'{HOST}/api/annotation_set/{ANNOTATION_SET_ID}/export?template=dataframe.json&disposition=attachment&include_columns=%5B"label.id"%2C"label.uuid"%2C"label.name"%2C"label.lineage_names"%2C"comment"%2C"needs_review"%2C"tag_names"%2C"updated_at"%2C"point.id"%2C"point.x"%2C"point.y"%2C"point.t"%2C"point.is_targeted"%2C"point.media.id"%2C"point.media.key"%2C"point.media.path_best"%2C"point.pose.timestamp"%2C"point.pose.lat"%2C"point.pose.lon"%2C"point.pose.alt"%2C"point.pose.dep"%2C"point.media.deployment.key"%2C"point.media.deployment.campaign.key"%5D&f=%7B"operations"%3A%5B%7B"module"%3A"pandas"%2C"method"%3A"json_normalize"%7D%2C%7B"method"%3A"sort_index"%2C"kwargs"%3A%7B"axis"%3A1%7D%7D%5D%7D&q=%7B"filters"%3A%5B%7B"name"%3A"label_id"%2C"op"%3A"is_not_null"%7D%5D%7D&translate=%7B"vocab_registry_keys"%3A%5B"worms"%2C+"caab"%2C+"catami"%5D%7D'
data = requests.get(url, headers={'X-auth-token': API_TOKEN}).json()  
annotations_data = requests.get(annotation_url, headers={'X-auth-token': API_TOKEN}).json() 
status_url = f"{HOST}{data['status_url']}"
result_url = f"{HOST}{data['result_url']}"
ann_status_url = f"{HOST}{annotations_data["status_url"]}"
ann_result_url = f"{HOST}{annotations_data["result_url"]}"
# Poll the task status until it completes
print("Polling task status...")
while True:
    status_response = requests.get(status_url, headers={'X-auth-token': API_TOKEN}).json()
    print("Task Status:", status_response)
    
    # Check if the task is complete
    if status_response.get("status") == "done":
        print("Task completed!")
        break
    elif status_response.get("status") == "error":
        print("Error in task processing:", status_response.get("error"))
        raise Exception("Task failed. Check the server response.")
    
    # Wait before polling again (e.g., 5 seconds)
    time.sleep(5)

# Retrieve the result
print("Fetching result...")
result_data = requests.get(result_url, headers={'X-auth-token': API_TOKEN}).json()

# create dir to save using name of media_collection
save_dir = os.path.join("media_collection", result_data.get("metadata",{}).get("name"))  
os.makedirs(save_dir, exist_ok=True) 


print("Polling task status...")
while True:
    ann_status_response = requests.get(ann_status_url, headers={'X-auth-token': API_TOKEN}).json()
    print("Task Status:", status_response)
    
    # Check if the task is complete
    if ann_status_response.get("status") == "done":
        print("Task completed!")
        break
    elif ann_status_response.get("status") == "error":
        print("Error in task processing:", ann_status_response.get("error"))
        raise Exception("Task failed. Check the server response.")
    
    # Wait before polling again (e.g., 5 seconds)
    time.sleep(5)

# Retrieve the result
print("Fetching result...")
ann_result_data = requests.get(ann_result_url, headers={'X-auth-token': API_TOKEN}).json()
with open('annotations/fish_annotations.json','w') as f:
    json.dump(ann_result_data, f)

# iterate through, download & save images
count = 0
for i in result_data.get("objects"):
    newpath = save_dir+i.get('path_best')[-32:]
    newpath = newpath.replace('\\', '/')
    file_path = Path(newpath)
    if not file_path.exists():
        print('Downloading New Picture')
        print(newpath)
        with open(os.path.join(save_dir, os.path.basename(i.get('path_best'))), 'wb') as handler:
            handler.write(requests.get(i.get('path_best')).content)             # define a get request using a specific endpoint
    else:
        print("Skipping Image - Already Downloaded")
print("DONE")

# Randomly select 326 images
#selected_images = random.sample(result_data.get("objects"), 326)

# # iterate through, download & save images
# count = 0
# for i in selected_images:
#     newpath = save_dir+i.get('path_best')[-32:]
#     newpath = newpath.replace('\\', '/')
#     file_path = Path(newpath)
#     if not file_path.exists():
#         print('Downloading New Picture')
#         print(newpath)
#         with open(os.path.join(save_dir, os.path.basename(i.get('path_best'))), 'wb') as handler:
#             handler.write(requests.get(i.get('path_best')).content)             # define a get request using a specific endpoint
#     else:
#         print("Skipping Image - Already Downloaded")
# print("DONE")


# ## Import annotations as json
