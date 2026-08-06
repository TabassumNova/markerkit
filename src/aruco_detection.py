import cv2
import argparse
import sys
import numpy as np


def serialize_corners_clockwise(corners):
    """Return 4 corners in clockwise order: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return pts.astype(np.int32)

    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered.astype(np.int32)


def get_effective_outer_corners(marker_data, fallback_mode="strict"):
    """Return usable outer corners while preserving raw outer_corners semantics.

    fallback_mode:
    - "strict": return None when outer_corners is missing.
    - "inner_as_outer": return inner_corners copy when outer_corners is missing.
    - "nan_sentinel": return 4x2 NaN array when outer_corners is missing.
    """
    outer = marker_data.get("outer_corners")
    if outer is not None:
        return outer.astype(np.float32), False

    inner = marker_data.get("inner_corners")
    if fallback_mode == "inner_as_outer" and inner is not None:
        return inner.astype(np.float32), True
    if fallback_mode == "nan_sentinel":
        return np.full((4, 2), np.nan, dtype=np.float32), True
    return None, True


def getAruco(image, aruco_dict_id, visualisation = True, debug=False):
    if image is None:
        print("Error: Could not load image.")
        sys.exit(1)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Default parameters
    aruco_params = cv2.aruco.DetectorParameters()

    print("[INFO] Detecting markers...")
    marker_dict = {}
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        print(f"[INFO] Detected {len(ids)} markers")
        print("Marker IDs:", ids.flatten().tolist())

        # Always detect border for each marker id and store both inner/outer corners.
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            inner_pts = serialize_corners_clockwise(marker_corners[0])
            outer_pts_raw = detect_white_border(marker_corners, image, debug=debug)
            outer_pts = serialize_corners_clockwise(outer_pts_raw) if outer_pts_raw is not None else None
            # Make sure that the inner and outer corners are serialized clockwise
            marker_dict[marker_id] = {
                "inner_corners": inner_pts.copy(),
                "outer_corners": outer_pts.copy() if outer_pts is not None else None,
                "outer_valid": outer_pts is not None,
                "outer_reason": "detected" if outer_pts is not None else "non_rectangular_or_not_found"
            }

        # refined_marker_dict = filter_marker_outliers(marker_dict, debug=debug)

        # Visualize only refined markers after outlier filtering.
        inner_vis = image.copy()
        outer_vis = image.copy()
        for marker_id, marker_data in marker_dict.items():
            inner_pts = marker_data["inner_corners"]
            outer_pts, is_fallback = get_effective_outer_corners(marker_data, fallback_mode="inner_as_outer")

            cv2.polylines(inner_vis, [inner_pts], True, (255, 255, 0), 2)
            for pt in inner_pts:
                cv2.circle(inner_vis, tuple(pt), 2, (0, 255, 255), -1)

            if outer_pts is not None:
                outer_pts_i32 = outer_pts.astype(np.int32)
                color = (0, 165, 255) if is_fallback else (0, 255, 0)
                cv2.polylines(outer_vis, [outer_pts_i32], True, color, 2)
                for pt in outer_pts_i32:
                    cv2.circle(outer_vis, tuple(pt), 2, (0, 0, 255), -1)

            cv2.putText(
                inner_vis,
                f"{marker_id}",
                tuple(inner_pts[0] + np.array([0, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )
            cv2.putText(
                outer_vis,
                f"{marker_id}",
                tuple(inner_pts[0] + np.array([0, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # cv2.imwrite('detected_inner_corners.jpg', inner_vis)
        # cv2.imwrite('detected_outer_corners.jpg', outer_vis)

        if visualisation:
            # Final visualization in separate windows.
            inner_display = cv2.resize(inner_vis, (960, max(540, int(h * 960 / w))))
            outer_display = cv2.resize(outer_vis, (960, max(540, int(h * 960 / w))))
            cv2.imshow('Detected ArUco Inner Corners', inner_display)
            cv2.imshow('Detected ArUco Outer Corners', outer_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("[INFO] No markers detected. Check image quality, lighting, or try custom dictionary. Rejected candidates:", len(rejected))
        marker_dict = {}

    return marker_dict


def filter_marker_outliers(marker_dict, distance_sigma=1.0, area_sigma=1.0, min_samples=3, debug=False):
    """Remove markers whose inner/outer relation is an outlier.

    Rules:
    1) Mean corner distance between inner and outer corners.
    2) Difference between outer and inner bounding-box areas.
    """
    metrics = []
    for marker_id, data in marker_dict.items():
        inner = data.get("inner_corners")
        outer = data.get("outer_corners")

        if inner is None or outer is None:
            continue
        if len(inner) != 4 or len(outer) != 4:
            continue

        inner_f = inner.astype(np.float32)
        outer_f = outer.astype(np.float32)

        mean_corner_distance = float(np.mean(np.linalg.norm(outer_f - inner_f, axis=1)))

        ix, iy, iw, ih = cv2.boundingRect(inner_f)
        ox, oy, ow, oh = cv2.boundingRect(outer_f)
        inner_bbox_area = float(iw * ih)
        outer_bbox_area = float(ow * oh)
        # bbox_area_gap = max(0.0, outer_bbox_area)

        # metrics.append((marker_id, mean_corner_distance, bbox_area_gap))
        metrics.append((marker_id, mean_corner_distance, outer_bbox_area))

    # Not enough markers to build stable mean/std thresholds.
    if len(metrics) < min_samples:
        if debug:
            print(f"Outlier check skipped (need >= {min_samples}, got {len(metrics)}).")
        return marker_dict

    dvals = np.array([m[1] for m in metrics], dtype=np.float32)
    avals = np.array([m[2] for m in metrics], dtype=np.float32)

    d_mean, d_std = float(np.mean(dvals)), float(np.std(dvals))
    a_mean, a_std = float(np.mean(avals)), float(np.std(avals))
    print(f"[INFO] Outlier filter: distance(mean={d_mean:.3f}, std={d_std:.3f}), outer_area(mean={a_mean:.3f}, std={a_std:.3f})")

    remove_ids = []
    for marker_id, dval, aval in metrics:
        print(f"[DEBUG] Marker ID {marker_id}: distance={dval:.3f}, outer_area={aval:.3f}")
        distance_outlier = (d_std > 0.0) and (abs(dval - d_mean) > distance_sigma * d_std)
        area_outlier = (a_std > 0.0) and (abs(aval - a_mean) > area_sigma * a_std)
        if distance_outlier or area_outlier:
            remove_ids.append(marker_id)
            if distance_outlier:
                print(f"[DEBUG] Marker ID {marker_id} flagged as distance outlier.")
            if area_outlier:
                print(f"[DEBUG] Marker ID {marker_id} flagged as area outlier.")

    for marker_id in remove_ids:
        # Keep marker entry and inner corners, but blank out invalid outer corners.
        if marker_id in marker_dict:
            marker_dict[marker_id]["outer_corners"] = None
            marker_dict[marker_id]["outer_valid"] = False
            marker_dict[marker_id]["outer_reason"] = "outlier"

    if debug:
        print(
            "[DEBUG] Outlier filter:",
            f"total={len(metrics)}",
            f"outer_corners_cleared={len(remove_ids)}",
            f"distance(mean={d_mean:.3f}, std={d_std:.3f})",
            f"bbox_gap(mean={a_mean:.3f}, std={a_std:.3f})"
        )
        if remove_ids:
            print(f"[DEBUG] Cleared outer corners for marker IDs: {sorted(remove_ids)}")

    return marker_dict


def find_large_contour(contours):
    # Find the largest contour considering the bounding box area
    best = None
    best_area = 0.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        if rect_area > best_area:
            best_area = rect_area
            best = c
    return best


def is_valid_outer_contour(contour, approx_eps_ratio=0.02, min_area_ratio=0.65, debug=False):
    """Check whether a contour is a plausible outer white border.

    The contour must:
    - occupy enough of its bounding box area to look reasonably 
    rectangular
    """

    contour_area = float(abs(cv2.contourArea(contour, True)))
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h)
    if bbox_area <= 0:
        return False

    area_ratio = contour_area / bbox_area
    if debug:
        print(
            f"[DEBUG] Contour validation: contour_area={contour_area:.2f}, "
            f"bbox_area={bbox_area:.2f}, ratio={area_ratio:.3f}"
        )

    return area_ratio >= min_area_ratio

def detect_white_border(corner, image, pad=20, morph_closing=False, debug=False, area_validation=False):
    """
    Detect border points for a single ArUco marker corner set.
    Args:
        corner: 4x2 array of inner corners (marker corners).
        image: Input image (BGR).
        pad: Padding around the marker bounding box for ROI extraction.
        morph_closing: Whether to apply morphological closing to the edge image.
        debug: Whether to show debug visualizations.
        area_validation: Whether to validate the detected contour based on its area ratio.
    Returns:
        4x2 array of outer corners (border points) if detected, otherwise None.
    
    """
    # Step 1: Extract the ROI with padding
    marker_corners = corner[0].astype(np.float32)
    x, y, w, h = cv2.boundingRect(marker_corners)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    roi = image[y0:y1, x0:x1].copy()

    if debug:
        cv2.imshow("White Border Debug - ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Step 2: Detect the edges
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    if morph_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Step 3: Detect the contours
    contour_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    if debug:
        contours_vis = roi.copy()
        cv2.drawContours(contours_vis, contours, -1, (0, 255, 255), 1)
        cv2.imshow("White Border Debug - Contours", contours_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not contours:
        return None

    # Step 4: Find best contour considering the bounding rect area
    best = find_large_contour(contours)
    if best is None:
        return None

    if debug:
        best_vis = roi.copy()
        cv2.drawContours(best_vis, [best], -1, (0, 0, 255), 2)
        cv2.imshow("White Border Debug - Best", best_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Filter contours that doesn't pass validation
    if area_validation and not is_valid_outer_contour(best, debug=debug):
        if debug:
            print("[DEBUG] Skipping outer border: contour failed quadrilateral/area validation.")
        return None

    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    outer_refined = box.astype(np.float32)
    outer_refined[:, 0] += x0
    outer_refined[:, 1] += y0

    if debug:
        # Box before global offset (ROI coordinates)
        box_roi = box.astype(np.int32)
        final_roi_vis = roi.copy()
        cv2.polylines(final_roi_vis, [box_roi], True, (255, 0, 0), 2)
        cv2.imshow("White Border Debug - Final Box ROI", final_roi_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return outer_refined.astype(np.int32)



if __name__ == "__main__":
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260805/FX10_obj10_pos_16_22_2026-08-05_11-38-25/FX10_obj10_pos_16_22_2026-08-05_11-38-25.png'
    image = cv2.imread(path)
    marker_dict = getAruco(image, aruco_dict_id=cv2.aruco.DICT_4X4_1000, debug=False)
    pass
