	t?????@t?????@!t?????@	?{????i??{????i?!?{????i?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$t?????@??<,Ԛ??AS??j??@Yt$???~??*	?????k@2F
Iterator::Model?q?????!0?%poV@)???~?:??1"q??U@:Preprocessing2j
3Iterator::Model::Prefetch::MapAndBatch::TensorSlice???{????!??I?!@)???{????1??I?!@:Preprocessing2]
&Iterator::Model::Prefetch::MapAndBatch??0?*x?!?v	ܻ?@)??0?*x?1?v	ܻ?@:Preprocessing2P
Iterator::Model::Prefetchn??t?!??^?V@)n??t?1??^?V@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?{????i?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??<,Ԛ????<,Ԛ??!??<,Ԛ??      ??!       "      ??!       *      ??!       2	S??j??@S??j??@!S??j??@:      ??!       B      ??!       J	t$???~??t$???~??!t$???~??R      ??!       Z	t$???~??t$???~??!t$???~??JCPU_ONLYY?{????i?b 