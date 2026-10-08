"""Restore omitted expression primitives from the matching full source GLB.

Usage: python tools/restore-avatar-expressions.py FULL_SOURCE.glb AVATAR.glb
Reuses the avatar's existing vertex pool, skeleton, texture atlas and animations.
"""
import copy
import json
from pathlib import Path
import struct
import sys


def read(path):
    data = Path(path).read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20:20 + length]), data[28 + length:]


def accessor(document, binary, index):
    a = document["accessors"][index]
    view = document["bufferViews"][a["bufferView"]]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    fmt = "<" + {5123: "H", 5125: "I", 5126: "f", 5121: "B"}[a["componentType"]] * components
    stride = view.get("byteStride", struct.calcsize(fmt))
    offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    return [struct.unpack_from(fmt, binary, offset + i * stride) for i in range(a["count"])]


def restore(source, target):
    full, fb = read(source)
    dest, db = read(target)
    source_prims = {full["materials"][p["material"]]["name"]: p for p in full["meshes"][0]["primitives"]}
    dest_prims = {dest["materials"][p["material"]]["name"]: p for p in dest["meshes"][0]["primitives"]}
    attrs = dest_prims["Head"]["attributes"]
    original_positions = accessor(full, fb, source_prims["Head"]["attributes"]["POSITION"])
    positions = accessor(dest, db, attrs["POSITION"])
    assert len(original_positions) == len(positions)
    max_error = max(abs(p[k] * (-.0085 if k == 0 else .0085) - q[k])
                    for p, q in zip(original_positions, positions) for k in range(3))
    assert max_error < .00002, f"Vertex pools do not match: {max_error}"
    original_indices = [v[0] for v in accessor(full, fb, source_prims["Head"]["indices"])]
    current_indices = [v[0] for v in accessor(dest, db, dest_prims["Head"]["indices"])]
    reversed_indices = [x for i in range(0, len(original_indices), 3)
                        for x in (original_indices[i], original_indices[i + 2], original_indices[i + 1])]
    reverse = original_indices != current_indices
    assert not reverse or reversed_indices == current_indices, "Unexpected topology remap"
    atlas = dest["materials"][dest_prims["Mouth_Base"]["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    binary = bytearray(db[:dest["buffers"][0]["byteLength"]])
    for name in ("Mouth_Playful", "Eye_Playful", "Mouth_Smile"):
        if name in dest_prims:
            continue
        primitive = source_prims[name]
        indices = [v[0] for v in accessor(full, fb, primitive["indices"])]
        if reverse:
            indices = [x for i in range(0, len(indices), 3) for x in (indices[i], indices[i + 2], indices[i + 1])]
        payload = struct.pack("<" + "H" * len(indices), *indices)
        binary.extend(b"\0" * (-len(binary) % 4))
        view_id = len(dest["bufferViews"])
        dest["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload), "target": 34963})
        binary.extend(payload)
        index_id = len(dest["accessors"])
        dest["accessors"].append({"bufferView": view_id, "componentType": 5123, "count": len(indices), "type": "SCALAR"})
        material = copy.deepcopy(full["materials"][primitive["material"]])
        material["pbrMetallicRoughness"]["baseColorTexture"]["index"] = atlas
        material_id = len(dest["materials"])
        dest["materials"].append(material)
        dest["meshes"][0]["primitives"].append({"attributes": attrs.copy(), "indices": index_id, "material": material_id, "mode": 4})
    for node in dest["nodes"]:
        events = node.get("extras", {}).get("submeshVisibilityEvents")
        if events:
            node["extras"]["submeshVisibilityEvents"] = {a["name"]: events[a["name"]] for a in dest["animations"]}
    dest["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(dest, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    Path(target).write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(encoded) + len(binary))
                            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
                            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    print("Restored original Mouth_Playful, Eye_Playful, Mouth_Smile; vertex match error:", max_error)


if __name__ == "__main__":
    restore(sys.argv[1], sys.argv[2])
