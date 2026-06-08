import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

hltVertexTable = cms.EDProducer("HLTVertexTableProducer",
                                skipNonExistingSrc = cms.bool(True),
                                pvSrc = cms.InputTag("hltOfflinePrimaryVertices"),
                                goodPvCut = cms.string("!isFake && ndof >= 4.0 && abs(z) <= 24.0 && abs(position.Rho) <= 2.0"), 
                                pfSrc = cms.InputTag("hltParticleFlowTmp"),
                                dlenMin = cms.double(0),
                                dlenSigMin = cms.double(3),
                                pvName = cms.string("hltPrimaryVertex"))

hltPixelVertexTable = cms.EDProducer("HLTVertexTableProducer",
                                     skipNonExistingSrc = cms.bool(True),
                                     pvSrc = cms.InputTag("hltPhase2PixelVertices"),
                                     goodPvCut = cms.string(""),
                                     usePF = cms.bool(False), # use directly the tracks from PV fit 
                                     pfSrc = cms.InputTag(""),
                                     dlenMin = cms.double(0),
                                     dlenSigMin = cms.double(3),
                                     pvName = cms.string("hltPixelVertex"))

hltSimVertexTable = cms.EDProducer("SimpleTrackingVertexFlatTableProducer",
                                         skipNonExistingSrc = cms.bool(False),
                                         src = cms.InputTag("mix", "MergedTrackTruth"),
                                         name = cms.string("hltSimVertex"),
                                         extension = cms.bool(False),
                                         variables = cms.PSet(
                                                              x   = Var("position().x()", float, doc = "secondary vertex X position, in cm",precision=10),
                                                              y   = Var("position().y()", float, doc = "secondary vertex Y position, in cm",precision=10),
                                                              z   = Var("position().z()", float, doc = "secondary vertex Z position, in cm",precision=14),
                                                              ntracks = Var("nDaughterTracks()", "uint8", doc = "number of tracks"),
                                                              bunchCrossing = Var("eventId().bunchCrossing()", int, doc = "bunch crossing"),
                                                              event = Var("eventId().event()", int, doc = "event number"),
                                                              ),
                                         )
