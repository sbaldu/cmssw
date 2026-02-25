import FWCore.ParameterSet.Config as cms

hltCLUEVertex = cms.EDProducer('CLUEVertexProducer@alpaka',
    Verbosity = cms.int32(0),
    PtMin = cms.double(1),
    Method2 = cms.bool(True),
    dc = cms.double(0.04),
    rhoc = cms.double(40.),
    dm = cms.double(0.04),
    seed_dc = cms.double(0.1),
    TrackCollection = cms.InputTag('hltPhase2PixelTracksSoA'),
    # beamSpot = cms.InputTag('offlineBeamSpot'),
    Finder = cms.string('DivisiveVertexFinder'),
    UseError = cms.bool(True),
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5),
    ZSeparation = cms.double(0.05),
    NTrkMin = cms.int32(2),
    PVcomparer = cms.PSet(
      track_pt_min = cms.double(1),
      track_pt_max = cms.double(10),
      track_chi2_max = cms.double(999999),
      track_prob_min = cms.double(-1)
    ),
    mightGet = cms.optional.untracked.vstring,
    alpaka = cms.untracked.PSet(
      backend = cms.untracked.string(''),
      synchronize = cms.optional.untracked.bool
    )
)

