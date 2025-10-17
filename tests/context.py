# -*- coding: utf-8 -*-

import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import spherical_spectral_element
from spherical_spectral_element.config import np