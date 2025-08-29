from solmoe.sim.lob import FillEngine, BookLevel


def test_fill_partial_and_queue():
    eng = FillEngine(latency_ms=50)
    levels = [BookLevel(price=10.0, size=5.0, ahead=3.0), BookLevel(price=10.01, size=2.0, ahead=0.0)]
    # Want to buy 4.0; at level 1, effective avail=2.0 (5-3), so remaining 2.0 goes to level 2
    filled = eng.compute_fill(side="BUY", size=4.0, levels=levels)
    assert abs(filled - 4.0) < 1e-9

    # If ahead >= avail, no fill at first level
    levels2 = [BookLevel(price=10.0, size=5.0, ahead=5.0), BookLevel(price=10.01, size=2.0, ahead=0.0)]
    filled2 = eng.compute_fill(side="BUY", size=3.0, levels=levels2)
    assert abs(filled2 - 2.0) < 1e-9

