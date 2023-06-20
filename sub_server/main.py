from fastapi import FastAPI, Response
from typing import Optional, Union
from oracleDB import oracleDB
from dotenv import load_dotenv
import os, json
from imageModelAI import ImageAI

load_dotenv()
app = FastAPI()
db = oracleDB(os.environ.get('ID'), os.environ.get('PW'), os.environ.get('IP'), os.environ.get('PORT'), os.environ.get('SID'))
        
def data(project_no: int) -> list:
    sql = f"select bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_data_type, bundle_folder_name from DataBundle where bundle_no = (select project_bundle_no from Labeling_Project where project_no = {project_no}) order by bundle_no"
    db.execute(sql)
    return db.get_DataBundles()

def labeling(project_no: int) -> list:
    sql = f"select data_no, worked_by, label from Labeling_Done where project_no = {project_no} order by project_no"    
    db.execute(sql)
    return db.get_Labeling_Done()

def label_counts(project_no: int) -> int:
    sql = f"select project_category from Labeling_Project where project_no = {project_no} order by project_no"
    db.execute(sql)
    return db.get_label_counts()

from fastapi.responses import FileResponse
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('main.py')

@app.get("/{project_no}", status_code=200)
def select(project_no: int) -> None:

    data_bundles = data(project_no)
    labeling_dones = labeling(project_no)
    labels = json.loads(label_counts(project_no)[0])['info'] # str을 json으로 바꾸고 'info'로 접속 -> list가 return
    print(data_bundles[0].bundle_uploaded_filename)
    print(labeling_dones[0].data_no)
    print(labels)
    if not data_bundles or not labeling_dones:
        return {'no': 'data'}
    
    ai = ImageAI()
    ai.save_img(data_bundles[0], labeling_dones)
    array = ai.convert_to_num(data_bundles[0]) # 이미지를 3차원 배열로 변환
    return ai.color(array, labeling_dones, labels)