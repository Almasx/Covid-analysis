	q????@q????@!q????@	?????$@?????$@!?????$@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q????@V-?????AH?}8}|@Ya??+?J@*	???????@2P
Iterator::Model::PrefetchTR'???J@!8zw5??X@)TR'???J@18zw5??X@:Preprocessing2F
Iterator::Model??6?J@!      Y@)?c?ZB??1""*???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?????$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V-?????V-?????!V-?????      ??!       "      ??!       *      ??!       2	H?}8}|@H?}8}|@!H?}8}|@:      ??!       B      ??!       J	a??+?J@a??+?J@!a??+?J@R      ??!       Z	a??+?J@a??+?J@!a??+?J@JCPU_ONLYY?????$@b 