	Qk?w?8?@Qk?w?8?@!Qk?w?8?@	R??CӨ+@R??CӨ+@!R??CӨ+@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Qk?w?8?@?QI??&??A?٬?\?{@Y<Nё\?Q@*	4333???@2P
Iterator::Model::Prefetch?:p??Q@!?'?X@)?:p??Q@1?'?X@:Preprocessing2F
Iterator::ModelȘ????Q@!      Y@)?J?4??1??@3®?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 13.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9R??CӨ+@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?QI??&???QI??&??!?QI??&??      ??!       "      ??!       *      ??!       2	?٬?\?{@?٬?\?{@!?٬?\?{@:      ??!       B      ??!       J	<Nё\?Q@<Nё\?Q@!<Nё\?Q@R      ??!       Z	<Nё\?Q@<Nё\?Q@!<Nё\?Q@JCPU_ONLYYR??CӨ+@b 