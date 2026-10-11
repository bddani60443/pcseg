class SweepCancelled(Exception):
    """
    Raised by a sweep (SegmentationService/AutoSegmentationService.run_sweep) when
    its `cancel_check` callback returns True between views - lets a caller (e.g. a
    GUI's Cancel button) distinguish a deliberate cancellation from a real error or
    a normal completion, rather than silently returning a partial result.
    """
