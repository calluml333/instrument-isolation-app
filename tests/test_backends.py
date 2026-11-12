import pytest

from isolate.models.base import Backend
from tests.contract import check_backend
from tests.fakes import IdentityBackend, LossyBackend

# Annotated so mypy checks the stand-ins structurally satisfy `Backend`.
STAND_INS: tuple[Backend, ...] = (IdentityBackend(), LossyBackend())


@pytest.mark.parametrize("backend", STAND_INS, ids=lambda b: b.name)
def test_stand_ins_meet_the_contract(backend):
    check_backend(backend)
