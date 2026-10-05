function vibratoCandidateVector = vibratoCandidates( ...
    vibratoIndicate, vibratoIndicateSign, frameCriterion, timeF)
% Compatibility implementation of AVA's intended passage grouping.
% It preserves AVA's >=6-frame rule while supporting zero or one passage.

    detectedFrames = find( ...
        vibratoIndicate == vibratoIndicateSign ...
    );
    detectedFrames = detectedFrames(:);
    timeF = timeF(:);

    if isempty(detectedFrames)
        vibratoCandidateVector = zeros(0, 2);
        return;
    end

    gapPositions = find(diff(detectedFrames) > 1);

    runStartPositions = [1; gapPositions + 1];
    runEndPositions = [gapPositions; numel(detectedFrames)];

    frameStarts = detectedFrames(runStartPositions);
    frameEnds = detectedFrames(runEndPositions);

    % AVA uses end-start >= 5, meaning at least six consecutive frames.
    keep = (frameEnds - frameStarts) >= frameCriterion;

    frameStarts = frameStarts(keep);
    frameEnds = frameEnds(keep);

    vibratoCandidateVector = [ ...
        timeF(frameStarts), ...
        timeF(frameEnds) ...
    ];
end