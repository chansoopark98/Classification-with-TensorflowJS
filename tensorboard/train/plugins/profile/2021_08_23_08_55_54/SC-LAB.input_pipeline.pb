  *i??|?=W@o=
ףx?@2U
Iterator::Model::PaddedBatchV2?m½2???!?E7?U@)?U??f???1?M?jU@:Preprocessing2?
?Iterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecords,??̷?!d|?X,?@)s,??̷?1d|?X,?@:Advanced file read2?
?Iterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4?pZ𢯨?!?F?X?b@)?pZ𢯨?1?F?X?b@:Preprocessing2?
nIterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::Prefetch::ParallelMapV2???U+??!< n?a??)???U+??1< n?a??:Preprocessing2}
FIterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2?&??d??!???_????)?&??d??1???_????:Preprocessing2?
UIterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2?˛õ??!??,e8??)?˛õ??1??,e8??:Preprocessing2n
7Iterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch??b??!!?W???)??b??1!?W???:Preprocessing2?
_Iterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::Prefetch??'???!??Jj??)??'???1??Jj??:Preprocessing2?
?Iterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::Prefetch::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap?lV}???!Z?}?1@)`=?[???1???)???:Preprocessing2?
?Iterator::Model::PaddedBatchV2::ForeverRepeat::Prefetch::ParallelMapV2::ParallelMapV2::Prefetch::ParallelMapV2::AssertCardinalitypD??k???!e???s
@)?Os?"??1Z?K?'??:Preprocessing2d
-Iterator::Model::PaddedBatchV2::ForeverRepeat?	?s3??!?,??
Q??)u??~?12???}???:Preprocessing2F
Iterator::Model?w?-;??!:???"?U@)
pu?10??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.