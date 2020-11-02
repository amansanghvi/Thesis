function pose = matchScanCustom(curr_points, ref_points)
    pose = matchScansGrid(lidarScan(curr_points), lidarScan(ref_points));
end