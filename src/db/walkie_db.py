from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Optional

import chromadb


@dataclass
class ObjectRecord:
    # define base object schema
    object_id: str
    object_xyz: Sequence[float]
    object_embedding: Sequence[float]
    heading: float
    scene_id: str | None = None
    # optional: from detectors that output class (e.g. YOLO)
    class_id: int | None = None
    class_name: str | None = None

@dataclass
class SceneRecord:
    # define base scene schema
    scene_id: str
    scene_xyz: Sequence[float]
    scene_embedding: Sequence[float]
    heading: float

@dataclass
class PersonRecord:
    # define base person schema
    person_id: str
    face_embedding: Sequence[float]
    person_name: str
    person_info: str
    
class WalkieVectorDB:
    def __init__(self, persist_directory: str = "chroma_db") -> None:
        # define local chroma client
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._objects_col = self._client.get_or_create_collection(
            name="objects",
            metadata={"hnsw:space": "cosine"},
        )
        self._scenes_col = self._client.get_or_create_collection(
            name="scenes",
            metadata={"hnsw:space": "cosine"},
        )
        self._people_col = self._client.get_or_create_collection(
            name="people",
            metadata={"hnsw:space": "cosine"},
        )
    
    # =========================================================================
    # Objects Collection Methods
    # =========================================================================

    def upsert_object(self, record: ObjectRecord) -> None:
        """Store or update an object with its embedding and coordinates."""
        xyz = list(record.object_xyz)
        if len(xyz) != 3:
            raise ValueError("object_xyz must have length 3")

        metadata = {
            "object_id": record.object_id,
            "object_x": float(xyz[0]),
            "object_y": float(xyz[1]),
            "object_z": float(xyz[2]),
            "heading": float(record.heading),
            "scene_id": record.scene_id or "",
        }
        if record.class_id is not None:
            metadata["class_id"] = int(record.class_id)
        if record.class_name is not None:
            metadata["class_name"] = str(record.class_name)

        self._objects_col.upsert(
            ids=[record.object_id],
            embeddings=[list(record.object_embedding)],
            metadatas=[metadata],
            documents=[""],
        )

    def delete_object(self, object_id: str) -> None:
        """Remove an object by ID."""
        self._objects_col.delete(ids=[object_id])

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve object metadata by ID."""
        result = self._objects_col.get(
            ids=[object_id],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas") or not result["metadatas"][0]:
            return None

        meta = dict(result["metadatas"][0])
        # Rebuild xyz for convenience
        meta["object_xyz"] = [
            meta.get("object_x"),
            meta.get("object_y"),
            meta.get("object_z"),
        ]
        return meta

    def query_objects(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        min_similarity: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Semantic search for objects by embedding similarity.
        Only returns hits with similarity >= min_similarity (cosine: 1 - distance).
        """
        result = self._objects_col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            distance = float(dist)
            similarity = 1.0 - distance  # cosine distance -> similarity
            if similarity < min_similarity:
                continue
            hit = dict(meta)
            hit["object_xyz"] = [
                hit.get("object_x"),
                hit.get("object_y"),
                hit.get("object_z"),
            ]
            hit["distance"] = distance
            hit["similarity"] = similarity
            hits.append(hit)

        return hits

    def find_nearby_object(
        self,
        class_id: int,
        position: tuple[float, float, float],
        radius: float = 1.0,
    ) -> Optional[str]:
        """Find an existing object with the same class_id within *radius* metres.

        Uses ChromaDB ``where`` filter to narrow by ``class_id``, then checks
        Euclidean distance on the stored (object_x, object_y, object_z) against
        the given *position*.

        Args:
            class_id: The YOLO class id to match.
            position: ``(x, y, z)`` of the newly detected object in metres.
            radius: Maximum Euclidean distance to consider a match (default 1.0 m).

        Returns:
            The ``object_id`` of the closest matching object, or ``None`` if no
            match is found within *radius*.
        """
        import math

        result = self._objects_col.get(
            where={"class_id": int(class_id)},
            include=["metadatas"],
        )

        if not result or not result.get("metadatas"):
            return None

        best_id: Optional[str] = None
        best_dist = float("inf")

        for meta in result["metadatas"]:
            ox = meta.get("object_x", 0.0)
            oy = meta.get("object_y", 0.0)
            oz = meta.get("object_z", 0.0)
            dist = math.sqrt(
                (position[0] - ox) ** 2
                + (position[1] - oy) ** 2
                + (position[2] - oz) ** 2
            )
            if dist < radius and dist < best_dist:
                best_dist = dist
                best_id = meta.get("object_id")

        return best_id

    def get_objects_by_scene(self, scene_id: str) -> List[Dict[str, Any]]:
        """Get all objects belonging to a specific scene."""
        result = self._objects_col.get(
            where={"scene_id": scene_id},
            include=["metadatas"],
        )

        if not result or not result.get("metadatas"):
            return []

        objects: List[Dict[str, Any]] = []
        for meta in result["metadatas"]:
            obj = dict(meta)
            obj["object_xyz"] = [
                obj.get("object_x"),
                obj.get("object_y"),
                obj.get("object_z"),
            ]
            objects.append(obj)

        return objects

    # =========================================================================
    # Scenes Collection Methods
    # =========================================================================

    def upsert_scene(self, record: SceneRecord) -> None:
        """Store or update a scene with its embedding and coordinates."""
        xyz = list(record.scene_xyz)
        if len(xyz) != 3:
            raise ValueError("scene_xyz must have length 3")

        metadata = {
            "scene_id": record.scene_id,
            "scene_x": float(xyz[0]),
            "scene_y": float(xyz[1]),
            "scene_z": float(xyz[2]),
            "heading": float(record.heading),
        }

        self._scenes_col.upsert(
            ids=[record.scene_id],
            embeddings=[list(record.scene_embedding)],
            metadatas=[metadata],
            documents=[""],
        )

    def delete_scene(self, scene_id: str) -> None:
        """Remove a scene by ID."""
        self._scenes_col.delete(ids=[scene_id])

    def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve scene metadata by ID."""
        result = self._scenes_col.get(
            ids=[scene_id],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas") or not result["metadatas"][0]:
            return None

        meta = dict(result["metadatas"][0])
        # Rebuild xyz for convenience
        meta["scene_xyz"] = [
            meta.get("scene_x"),
            meta.get("scene_y"),
            meta.get("scene_z"),
        ]
        return meta

    def query_scenes(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        min_similarity: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Semantic search for scenes by embedding similarity.
        Only returns hits with similarity >= min_similarity (cosine: 1 - distance).
        """
        result = self._scenes_col.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            distance = float(dist)
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            hit = dict(meta)
            hit["scene_xyz"] = [
                hit.get("scene_x"),
                hit.get("scene_y"),
                hit.get("scene_z"),
            ]
            hit["distance"] = distance
            hit["similarity"] = similarity
            hits.append(hit)

        return hits

    # =========================================================================
    # People Collection Methods
    # =========================================================================

    def upsert_person(self, record: PersonRecord) -> None:
        """Store or update a person with their face embedding and info."""
        metadata = {
            "person_id": record.person_id,
            "person_name": record.person_name,
            "person_info": record.person_info,
        }

        self._people_col.upsert(
            ids=[record.person_id],
            embeddings=[list(record.face_embedding)],
            metadatas=[metadata],
            documents=[""],
        )

    def delete_person(self, person_id: str) -> None:
        """Remove a person by ID."""
        self._people_col.delete(ids=[person_id])

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve person metadata by ID."""
        result = self._people_col.get(
            ids=[person_id],
            include=["metadatas"],
        )

        if not result or not result.get("metadatas") or not result["metadatas"][0]:
            return None

        return dict(result["metadatas"][0])

    def query_people(
        self,
        face_embedding: Sequence[float],
        n_results: int = 5,
        min_similarity: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Face matching by embedding similarity.
        Only returns hits with similarity >= min_similarity (cosine: 1 - distance).
        """
        result = self._people_col.query(
            query_embeddings=[list(face_embedding)],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas_batch = result.get("metadatas", [[]])[0]
        distances_batch = result.get("distances", [[]])[0]

        hits: List[Dict[str, Any]] = []
        for meta, dist in zip(metadatas_batch, distances_batch):
            distance = float(dist)
            similarity = 1.0 - distance
            if similarity < min_similarity:
                continue
            hit = dict(meta)
            hit["distance"] = distance
            hit["similarity"] = similarity
            hits.append(hit)

        return hits

    def delete_all_objects(self) -> None:
        """Delete all objects."""
        self._client.delete_collection("objects")
        # Create a new collection
        self._objects_col = self._client.get_or_create_collection(
            name="objects",
            metadata={"hnsw:space": "cosine"},
        )

    def delete_all_scenes(self) -> None:
        """Delete all scenes."""
        self._client.delete_collection("scenes")
        # Create a new collection
        self._scenes_col = self._client.get_or_create_collection(
            name="scenes",
            metadata={"hnsw:space": "cosine"},
        )
    def delete_all_people(self) -> None:
        """Delete all people."""
        self._client.delete_collection("people")
        # Create a new collection
        self._people_col = self._client.get_or_create_collection(
            name="people",
            metadata={"hnsw:space": "cosine"},
        )
    
    def delete_all(self) -> None:
        """Delete all objects, scenes, and people."""
        self.delete_all_objects()
        self.delete_all_scenes()
        self.delete_all_people()

__all__ = ["ObjectRecord", "SceneRecord", "PersonRecord", "WalkieVectorDB"]
