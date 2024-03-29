function [pose, cov, score] = matchScanCustom(...
    curr_points, ...  % The current lidar scan points
    ref_points, ...   % The points in the reference map
    pose_guess, ...   % initial guess for the pose [x, y, theta]
    cells_per_m, ...  % number of cells per m to use as resolution
    pose_range ...    % [x_range, y_range, th_range] such that the match will be within
    ...               % (pose_guess - pose_range, pose_guess + pose_range)
)
    [pose, stats] = matchScansGrid(...
        lidarScan(curr_points),... 
        lidarScan(ref_points), ...
        'MaxRange', 15, ...
        'InitialPose', pose_guess, ...
        'Resolution', cells_per_m, ...
        'TranslationSearchRange', pose_range(1:2), ...
        'RotationSearchRange', pose_range(3) ...
    );
    
    if (isValidPose(pose, pose_guess, pose_range))
       cov = stats.Covariance;
       score = stats.Score;
%        if (score/length(curr_points) > 0.5)
%            return
%        end
    else
        score = 0;
        cov = NaN(3);
        return
    end
    
    % Then use this one.
    [new_pose, stats] = matchScans( ...
        lidarScan(curr_points),... 
        lidarScan(ref_points), ...
        'InitialPose', pose, ...
        'MaxIterations', 500, ...
        'CellSize', 0.1);
    if (isValidPose(new_pose, pose_guess, pose_range))
        if (stats.Score*2 > score)
            score = stats.Score;
            pose = new_pose;
        else
            return
        end
    elseif(isValidPose(pose, pose_guess, pose_range))
        return
    else
        score = 0;
        cov = NaN(3);
    end

    function isValid = isValidPose(pose, pose_guess, pose_range)
        isTranslationValid = all(abs(pose(1:2) - pose_guess(1:2)) < abs(pose_range(1:2)));
        isRotationValid = abs(wrapToPi(pose(3) - pose_guess(3))) < abs(pose_range(3));
        isPoseValid = any(pose ~= 0);
        isValid = isTranslationValid && isRotationValid && isPoseValid;
    end
end