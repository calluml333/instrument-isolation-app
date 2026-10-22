import socket

import pytest
from pytest_socket import SocketBlockedError


# pytest-socket warns whenever it blocks a socket; here that is the point.
@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_network_is_blocked():
    # Tests must never reach the network (e.g. to download model weights).
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
