
import FWCore.ParameterSet.Config as cms
from ..modules.hltCLUEVertex_cfi import hltCLUEVertex

HLTCLUEVertexSequence = cms.Sequence(hltCLUEVertex)
