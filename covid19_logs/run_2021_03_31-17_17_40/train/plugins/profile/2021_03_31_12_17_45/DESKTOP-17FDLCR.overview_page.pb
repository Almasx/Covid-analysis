?	??ʡE??@??ʡE??@!??ʡE??@	_WE???(@_WE???(@!_WE???(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ʡE??@?[ A???A??%䃔}@Y?/?$?P@*333?4?:A)      ?=2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorX?2?1??@!y?"RX@)X?2?1??@1y?"RX@:Preprocessing2P
Iterator::Model::Prefetch'?????P@!<DL1@)'?????P@1<DL1@:Preprocessing2F
Iterator::Model?5^?I?P@!????_5@)U???N@??1????xa?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?D??4??@!???UX@)a??+ei?1Z???'?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9_WE???(@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?[ A????[ A???!?[ A???      ??!       "      ??!       *      ??!       2	??%䃔}@??%䃔}@!??%䃔}@:      ??!       B      ??!       J	?/?$?P@?/?$?P@!?/?$?P@R      ??!       Z	?/?$?P@?/?$?P@!?/?$?P@JCPU_ONLYY_WE???(@b Y      Y@qL8?Ri???"?
both?Your program is MODERATELY input-bound because 12.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 