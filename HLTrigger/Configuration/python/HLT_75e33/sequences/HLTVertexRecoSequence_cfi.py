import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.alpaka_cff import alpaka


from ..modules.hltOfflinePrimaryVertices_cfi import *
from ..modules.hltTrackRefsForJetsBeforeSorting_cfi import *
from ..modules.hltTrackWithVertexRefSelectorBeforeSorting_cfi import *
from ..modules.hltUnsortedOfflinePrimaryVertices_cfi import *
from ..sequences.HLTInitialStepPVSequence_cfi import *
from ..modules.clueVertexProducer_cfi import *

HLTVertexRecoSequence = cms.Sequence(HLTInitialStepPVSequence+hltUnsortedOfflinePrimaryVertices+hltTrackWithVertexRefSelectorBeforeSorting+hltTrackRefsForJetsBeforeSorting+hltOfflinePrimaryVertices+clueVertexProducer)
