.. raw:: html

  <style>
    .header-article { height: 0% !important;}
	.header-article-main { height: 0% !important;}
	.header-article__left { height: 0% !important;}
	.header-article__right { height: 0% !important;}
	.headerbtn { display: none !important;}
  </style>

WaveSpin Documentation
=======================

.. image:: _images/spin_both.gif
  :width: 600px
  :alt: WaveSpin animation

Joint Time-Frequency Scattering, Wavelet Scattering: features for classification, regression, and synthesis of audio, biomedical, and other signals. Friendly overviews:

  - `Wavelet Scattering <https://dsp.stackexchange.com/a/78513/50076>`_
  - `Joint Time-Frequency Scattering <https://dsp.stackexchange.com/a/78623/50076>`_
	
For benchmarks and main overview, see `GitHub repository <https://github.com/gptanon/wavespon>`_.

Installation
------------

`pip install wavespin`. Or, for latest version (most likely stable):

`pip install git+https://github.com/OverLordGoldDragon/wavespin`

Examples
--------

.. include:: _examples_gallery.txt

More examples
-------------

See `Examples <examples-rendered/index.html>`_.


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Core Documentation
   
   examples-rendered/index
   Scattering Docs <scattering_docs>
   hows_it_work

.. toctree::
   :maxdepth: 2
   :caption: Extended
   
   extended-docs/performance_tips
   extended-docs/general_method_docs
   extended-docs/further_reading
   API Reference <wavespin>
