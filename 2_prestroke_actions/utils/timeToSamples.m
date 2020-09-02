function samples = timeToSamples(time, offset, samplingRate)
    samples = (time - offset) / 1000 * samplingRate;
end