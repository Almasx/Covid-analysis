	q=
ףA?@q=
ףA?@!q=
ףA?@	?/O?*s#@?/O?*s#@!?/O?*s#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$q=
ףA?@?O??e??At???V}@Y]?FxKI@*	????y??@2P
Iterator::Model::Prefetch?7??dBI@!{?7?J?X@)?7??dBI@1{?7?J?X@:Preprocessing2F
Iterator::Model?0?*HI@!      Y@),e?X??1z6 oն?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?/O?*s#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?O??e???O??e??!?O??e??      ??!       "      ??!       *      ??!       2	t???V}@t???V}@!t???V}@:      ??!       B      ??!       J	]?FxKI@]?FxKI@!]?FxKI@R      ??!       Z	]?FxKI@]?FxKI@!]?FxKI@JCPU_ONLYY?/O?*s#@b 