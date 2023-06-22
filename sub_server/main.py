from fastapi import FastAPI, Response
from typing import Optional, Union
from oracleDB import oracleDB
from dotenv import load_dotenv
import os, json
from imageModelAI import ImageAI
from fastapi.responses import FileResponse

load_dotenv()
app = FastAPI()
db = oracleDB(os.environ.get('ID'), os.environ.get('PW'), os.environ.get('IP'), os.environ.get('PORT'), os.environ.get('SID'))

@app.get('/favicon.ico', include_in_schema=False)
async def favicon() -> None:
    return FileResponse('main.py')

@app.get("/labeling.result/{project_no}", status_code=200)
def select(project_no: int) -> None:
    try:
        db.learning_start(project_no)
        data_bundles = db.data(project_no)
        labeling_dones = db.labeling(project_no)
        labels = json.loads(db.label_names(project_no)[0])['info'] # str을 json으로 바꾸고 'info'로 접속 -> [라벨명, 라벨명, ...]이 return
        if not data_bundles or not labeling_dones:
            return {'no': 'data'}
        
        ai = ImageAI()
        (array, originals) = ai.save_img(project_no, data_bundles[0], labeling_dones) # [image파일명, 작업자, 고른 답]을 return
        convert_images_3D = ai.convert_to_num(project_no, data_bundles[0]) # 이미지를 3차원 배열로 변환
        res = ai.color(project_no, array, originals, convert_images_3D, labels)
        db.learning_end(project_no)
        return res
    except:
        db.learning_rollback(project_no)
        return {'error': 'occurred'}