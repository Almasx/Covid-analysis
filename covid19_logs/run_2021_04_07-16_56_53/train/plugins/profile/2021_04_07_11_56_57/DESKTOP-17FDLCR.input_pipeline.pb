	4??7?F?@4??7?F?@!4??7?F?@	R?0?A@R?0?A@!R?0?A@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$4??7?F?@B`??"???A=,Ԛ?|@Y???&?l@*	gfff?FA2P
Iterator::Model::Prefetch8gDi?l@!???$C?X@)8gDi?l@1???$C?X@:Preprocessing2F
Iterator::Modelё\?C?l@!      Y@)z6?>W??1?tlHb???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9S?0?A@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	B`??"???B`??"???!B`??"???      ??!       "      ??!       *      ??!       2	=,Ԛ?|@=,Ԛ?|@!=,Ԛ?|@:      ??!       B      ??!       J	???&?l@???&?l@!???&?l@R      ??!       Z	???&?l@???&?l@!???&?l@JCPU_ONLYYS?0?A@b 