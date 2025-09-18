import logging
from najaeda import naja

def constructDFF(lib):
  cell = naja.SNLDesign.createPrimitive(lib, "dff")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Output, "Q")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "RN")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "SN")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "CK")
  naja.SNLScalarTerm.create(cell, naja.SNLTerm.Direction.Input, "D")

def load(db):
  logging.info("Loading verilog built-in + iccad primitives")
  lib = naja.NLLibrary.createPrimitives(db, "iccad")
  constructDFF(lib)
