# V23D — Orbit Video to Textured 3D Human

End-to-end pipeline: input an orbit video of a person → get a textured 3D SMPL mesh ready for Unity.

---

## What you need

| Input | Description |
|---|---|
| Orbit video (`.mp4`) | 360° video of the person |
| SMPL model PKL | `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) |
| SMPL rigged FBX (for Unity) | `SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx` from the same site |

Python environment: `E:\envs\v23d_local\python.exe`

---

## Step 1 — Extract frames from the orbit video

```powershell
E:\envs\v23d_local\python.exe C:\V23D\V23D\workflows\pipeline_orchestration\parse_video.py `
    --video  "E:\my_orbit_video.mp4" `
    --out    "E:\zero123_dataset\humans_train\person_017" `
    --fps    2
```

This saves one frame every 0.5 seconds into the output folder.  
- **Frame 000** = front view → use as `FRONT_REF_IMG`  
- **Frame ~165** (roughly halfway/back) = back view → use as `BACK_REF_IMG`

---

## Step 2 — Fit SMPL mesh to the front frame

Run SMPLify-X on the front frame to produce the SMPL mesh (betas = body shape, thetas = pose):

```powershell
E:\envs\v23d_local\python.exe C:\V23D\V23D\third_party\smplify-x\smplifyx\main.py `
    --config        C:\V23D\V23D\third_party\smplify-x\cfg_files\fit_smpl.yaml `
    --data_folder   "E:\zero123_dataset\humans_train\person_017\frame_000" `
    --output_folder "E:\V23D_Data\smplifyx_output" `
    --model_folder  "E:\SMPL_extracted\SMPL_python_v.1.1.0\smpl\models"
```

This outputs the fitted mesh as `E:\smpl_textured_from_splat.obj`.

---

## Step 3 — Bake texture onto the mesh

Set these three paths at the top of `project_face_body_dual_view.py` to match your extracted frames:

```python
FRONT_REF_IMG = Path(r"E:/zero123_dataset/humans_train/person_017/frame_000/reference.png")
BACK_REF_IMG  = Path(r"E:/zero123_dataset/humans_train/person_017/frame_165/reference.png")
OBJ_PATH      = Path(r"E:/smpl_textured_from_splat.obj")
```

Then run:

```powershell
E:\envs\v23d_local\python.exe C:\V23D\project_face_body_dual_view.py
```

**Outputs:**

| File | Description |
|---|---|
| `E:\smpl_textured_face_body_dualview.ply` | Final textured mesh |
| `E:\unity_vertex_match_output\our_pipeline_vertex_match.obj` | UV-mapped mesh for Unity |
| `E:\unity_vertex_match_output\our_pipeline_vertex_match_albedo.png` | 4096×4096 texture atlas |

---

## Step 4 — Rebake texture for the rigged FBX

The SMPL FBX has a different UV layout from the OBJ. This converts the texture to match it:

```powershell
E:\envs\v23d_local\python.exe C:\V23D\rebake_texture_to_fbx_uv.py `
    --source_obj     "E:\unity_vertex_match_output\our_pipeline_vertex_match.obj" `
    --source_texture "E:\unity_vertex_match_output\our_pipeline_vertex_match_albedo.png" `
    --target_fbx     "\\students\student-n-r\rgdix\Downloads\SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx" `
    --output_texture "E:\unity_vertex_match_output\SMPL_m_unityDoubleBlends_rebaked_albedo.png"
```

**Final texture:**
```
E:\unity_vertex_match_output\SMPL_m_unityDoubleBlends_rebaked_albedo.png
```

---

## Unity import

1. Drag `SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx` and `SMPL_m_unityDoubleBlends_rebaked_albedo.png` into your Unity `Assets/` folder.
2. Select the FBX → **Materials** tab → **Extract Materials**.
3. Open the extracted material → shader set to **Standard** or **URP/Lit**.
4. Drag the PNG onto **Base Map / Albedo**.

The model will have your person's texture and retain all skeleton joints for animation.

---

## Optional flags

Edit at the top of `project_face_body_dual_view.py`:

| Flag | Default | Effect |
|---|---|---|
| `APPLY_FACE_HAIR_TEXTURE` | `False` | Also texture the face/hair |
| `APPLY_ARM_TEXTURE` | `True` | Texture the arms |
| `CHEST_DETECTOR_MODE` | `"yolo"` | Chest detector: `"yolo"` or `"mediapipe"` |
