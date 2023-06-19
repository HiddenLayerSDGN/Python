from fastapi import FastAPI, Response
from typing import Optional, Union
from oracleDB import oracleDB
from dotenv import load_dotenv
import os
from imageModelAI import ImageAI

load_dotenv()
app = FastAPI()
db = oracleDB(os.environ.get('ID'), os.environ.get('PW'), os.environ.get('IP'), os.environ.get('PORT'), os.environ.get('SID'))
        
def data(project_no: int):
    sql = f"select bundle_no, bundle_uploader, bundle_uploaded_filename, bundle_folder_name from DataBundle where bundle_no = (select project_bundle_no from Labeling_Project where project_no = {project_no}) order by bundle_no"
    db.execute(sql)
    data_bundles = db.get_DataBundles()
    return data_bundles

def labeling(project_no: int):
    sql = f"select data_no, worked_by, label from Labeling_Done where project_no = {project_no} order by project_no"
    db.execute(sql)
    labeling_dones = db.get_Labeling_Done()
    return labeling_dones

@app.get("/{table}")
def select(table: str, project_no: Union[int, None] = None) -> None:
    # if table != 'favicon.ico': # 강제 요청 때문에 만든 코드...
        # return Response('<link rel="icon" href="data:,">')
    data_bundles = data(project_no)
    labeling_dones = labeling(project_no)
    
    # return data_bundles
    # return labeling_dones

    ai = ImageAI()
    ai.save_img(data_bundles[0], labeling_dones)
    # array = ai.convert_to_num(data_bundles[0]) # 이미지가 3차원 배열이 됨