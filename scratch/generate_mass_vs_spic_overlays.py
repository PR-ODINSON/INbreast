import os
import glob
import re
import plistlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

base_dir = r"D:\Avantika\INbreast_complete"
xml_dir = os.path.join(base_dir, "AllXML")
dicom_dir = os.path.join(base_dir, "AllDICOMs")
out_dir = r"D:\Avantika\debug_overlays\spiculated_vs_mass"
os.makedirs(out_dir, exist_ok=True)

images_with_both = ['20586934', '20586986', '20587612', '20587664', '20587994', '20588046', '22613770', '22670094', '22670147', '22670465']

for idx, img_id in enumerate(images_with_both[:10]):
    xml_path = os.path.join(xml_dir, f"{img_id}.xml")
    dcm_candidates = glob.glob(os.path.join(dicom_dir, f"{img_id}_*.dcm"))
    if not dcm_candidates:
        continue
    dcm_path = dcm_candidates[0]
    
    # Read DICOM using basic parser or OpenCV/pydicom if available
    # Since pydicom might not be in this python env, let's read the preprocessed image from Preprocessed_complete if available
    prep_img_path = os.path.join(base_dir, "Preprocessed_complete", "images", f"{img_id}.png")
    
    with open(xml_path, "rb") as f:
        pl = plistlib.load(f)
        
    mass_polys = []
    spic_polys = []
    
    for img in pl.get("Images", []):
        for roi in img.get("ROIs", []):
            name = str(roi.get("Name", "")).strip().lower()
            pts_str = roi.get("Point_px", [])
            poly = []
            for pt in pts_str:
                m = re.match(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", pt)
                if m:
                    poly.append([float(m.group(1)), float(m.group(2))])
            if "mass" == name:
                mass_polys.append(poly)
            elif "spiculat" in name or "espiculated" in name:
                spic_polys.append(poly)
                
    # Create high-res mask overlays
    # Full res is 4084 x 3328
    H, W = 4084, 3328
    mask_mass = np.zeros((H, W), dtype=np.uint8)
    mask_spic = np.zeros((H, W), dtype=np.uint8)
    
    for poly in mass_polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_mass, [pts], 1)
        
    for poly in spic_polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_spic, [pts], 1)
        
    # Find bounding box around both to zoom in
    combined = mask_mass | mask_spic
    ys, xs = np.where(combined > 0)
    if len(ys) == 0:
        continue
    ymin, ymax = max(0, ys.min() - 100), min(H, ys.max() + 100)
    xmin, xmax = max(0, xs.min() - 100), min(W, xs.max() + 100)
    
    crop_mass = mask_mass[ymin:ymax, xmin:xmax]
    crop_spic = mask_spic[ymin:ymax, xmin:xmax]
    
    # Plot side by side
    fig, ax = plt.subplots(figsize=(8, 8))
    # Display background
    ax.imshow(np.zeros_like(crop_mass), cmap="gray", vmin=0, vmax=1)
    
    # Contours
    cnts_spic, _ = cv2.findContours(crop_spic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_mass, _ = cv2.findContours(crop_mass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    canvas = np.zeros((crop_mass.shape[0], crop_mass.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(canvas, cnts_spic, -1, (255, 50, 50), 3)  # Spiculated in Red
    cv2.drawContours(canvas, cnts_mass, -1, (50, 255, 50), 3)  # Mass in Bright Green
    
    ax.imshow(canvas)
    ax.set_title(f"Image {img_id}: Precise Mass (Green) vs Coarse Ellipse Spiculated Region (Red)\nArea: Mass={cv2.contourArea(cnts_mass[0]) if cnts_mass else 0:.0f} px, Spiculated={cv2.contourArea(cnts_spic[0]) if cnts_spic else 0:.0f} px", fontsize=10)
    ax.axis("off")
    
    save_path = os.path.join(out_dir, f"{idx+1:02d}_{img_id}_mass_vs_spiculated.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

print(f"Generated 10 comparison overlays in {out_dir}")

