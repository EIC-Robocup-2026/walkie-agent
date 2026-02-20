import uuid
from dotenv import load_dotenv

load_dotenv()

from src.db import WalkieVectorDB, ObjectRecord, SceneRecord
from src.vision import WalkieVision


def main():
    print("Mocking data for WalkieVectorDB")
    print("=" * 60)

    vision = WalkieVision(
        caption_provider="paligemma",
        embedding_provider="clip",
        detection_provider="yolo",
        preload=True,
    )
    
    # 2. Upsert into DB
    db = WalkieVectorDB()
    db.delete_all()
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"
    
    objects = [
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[0.000, 0.000, 0.000],
            object_embedding=vision.embed_text("start point"),
            heading=0,
            scene_id=scene_id,
            class_id=1,
            class_name="Start Point",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[0.525, -1.637, 0.000],
            object_embedding=vision.embed_text("table"),
            heading=-3.11928,
            scene_id=scene_id,
            class_id=2,
            class_name="Table",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[2.3287, -0.4838, 0.000],
            object_embedding=vision.embed_text("person"),
            heading=-1.33539,
            scene_id=scene_id,
            class_id=2,
            class_name="Person",
        )
    ]
    
    for obj in objects:
        db.upsert_object(obj)
    
    query_result = db.query_objects(vision.embed_text("water bottle"))
    print(query_result)


if __name__ == "__main__":
    main()