import uuid
from dotenv import load_dotenv

load_dotenv()

from src.db import WalkieVectorDB, ObjectRecord, SceneRecord
from src.vision import WalkieVision


def main():
    print("Mocking data for WalkieVectorDB")
    print("=" * 60)

    vision = WalkieVision(
        caption_provider="google",
        embedding_provider="clip",
        detection_provider="yolo",
    )
    vision.open()
    
    # 2. Upsert into DB
    db = WalkieVectorDB()
    db.delete_all()
    scene_id = f"scene_{uuid.uuid4().hex[:8]}"
    
    objects = [
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[1.0, 0.5, 0.0],
            object_embedding=vision.embed_text("water bottle"),
            heading=0.0,
            scene_id=scene_id,
            class_id=1,
            class_name="water bottle",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[-1.0, 0.0, 0.0],
            object_embedding=vision.embed_text("person"),
            heading=0.0,
            scene_id=scene_id,
            class_id=2,
            class_name="person",
        ),
        ObjectRecord(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            object_xyz=[0.0, 0.0, 0.0],
            object_embedding=vision.embed_text("chair"),
            heading=0.0,
            scene_id=scene_id,
            class_id=3,
            class_name="chair",
        ),
    ]
    
    for obj in objects:
        db.upsert_object(obj)
    
    query_result = db.query_objects(vision.embed_text("water bottle"))
    print(query_result)


if __name__ == "__main__":
    main()