function vibratoParameters = getVibratoParaFDM( ...
    vibratoIndicate, vibratoIndicateSign, frameCriterion, FDMoutput)
% Compatibility implementation of AVA's intended parameter aggregation.
% It preserves AVA's frame criterion and arithmetic mean calculation.

    detectedFrames = find( ...
        vibratoIndicate == vibratoIndicateSign ...
    );
    detectedFrames = detectedFrames(:);

    if isempty(detectedFrames)
        vibratoParameters = zeros(0, 2);
        return;
    end

    gapPositions = find(diff(detectedFrames) > 1);

    runStartPositions = [1; gapPositions + 1];
    runEndPositions = [gapPositions; numel(detectedFrames)];

    frameStarts = detectedFrames(runStartPositions);
    frameEnds = detectedFrames(runEndPositions);

    keep = (frameEnds - frameStarts) >= frameCriterion;

    frameStarts = frameStarts(keep);
    frameEnds = frameEnds(keep);

    vibratoParameters = zeros(numel(frameStarts), 2);

    for index = 1:numel(frameStarts)
        vibratoParameters(index, :) = mean( ...
            FDMoutput(frameStarts(index):frameEnds(index), :), ...
            1 ...
        );
    end
end