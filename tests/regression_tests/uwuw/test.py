import openmc
import openmc.capi
from openmc.stats import Box
from openmc.material import Materials

import pytest
from tests.testing_harness import PyAPITestHarness

pytestmark = pytest.mark.skipif(
    not openmc.capi._dagmc_enabled(),
    reason="DAGMC CAD geometry is not enabled.")

class UWUWTest(PyAPITestHarness):

    def _build_inputs(self):
        model = openmc.model.Model()

        # settings
        model.settings.batches = 5
        model.settings.inactive = 0
        model.settings.particles = 100

        source = openmc.Source(space=Box([-4, -4, -4],
                                         [ 4,  4,  4]))
        model.settings.source = source

        model.settings.dagmc = True

        model.settings.export_to_xml()

        # tally
        tally = openmc.Tally()
        tally.scores = ['total']
        tally.filters = [openmc.CellFilter(1)]
        model.tallies = [tally]

        model.tallies.export_to_xml()

def test_uwuw():
    harness = UWUWTest('statepoint.5.h5')
    harness.main()
