jackhmmer.sh <sequenceFile> <fastaDatabase>
python convert_to_proteinnet.py <sequenceFile>
python convert_to_tfrecord.py <sequenceFile>.proteinnet <sequenceFile>.tfrecord 42
cp <sequenceFile>.tfrecord <baseDirectory>/data/<datasetName>/testing
python protling.py <baseDirectory>/runs/<runName>/<datasetName>/<configurationFile> -d [baseDirectory] -p -e weighted_testing