?	n??~@n??~@!n??~@	?wc@2#@?wc@2#@!?wc@2#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$n??~@z?):????Ac?ZB>u|@Y?~?:p.H@*	3333?a<A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator[B>??N?@!?>?ͳYX@)[B>??N?@1?>?ͳYX@:Preprocessing2P
Iterator::Model::Prefetch?~?:p&H@!????@)?~?:p&H@1????@:Preprocessing2F
Iterator::ModelR???)H@!?U?@)_?Qڛ?1]???u?W?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap,Ԛ??N?@!?OR=?YX@)?q????o?1 v??p{+?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9?wc@2#@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z?):????z?):????!z?):????      ??!       "      ??!       *      ??!       2	c?ZB>u|@c?ZB>u|@!c?ZB>u|@:      ??!       B      ??!       J	?~?:p.H@?~?:p.H@!?~?:p.H@R      ??!       Z	?~?:p.H@?~?:p.H@!?~?:p.H@JCPU_ONLYY?wc@2#@b Y      Y@q?I??? ??"?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 