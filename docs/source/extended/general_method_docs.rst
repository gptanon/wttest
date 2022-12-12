General/Method Docs
*******************

`'width'` vs `'support'`
------------------------

.. collapse:: code

  .. code-block:: python
  
    import numpy as np
    import matplotlib.pyplot as plt
    
    #%% Support vs Width visuals #################################################
    t = np.linspace(-1, 1, 512)
    
    g0 = np.exp(-t**2 * 6)
    g1 = np.zeros(len(t))
    num = int(len(t)/3)
    g1[num:-num] = np.cos(3.1*(np.linspace(0, 1, len(t)-2*num) -.5))**(1/8)
    g2 = np.exp(-np.abs(t) * 20)
    ramp = np.linspace(np.sqrt(.01), np.sqrt(.04), len(t)//2)**2
    g2 += np.hstack([ramp, ramp[::-1]])
    g3 = np.exp(-np.abs(t) * 16)
    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    tkw = dict(loc='left', weight='bold', fontsize=17)
    axes[0, 0].plot(g0)
    axes[0, 0].set_title("high width, high support", **tkw)
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 1].plot(g1)
    axes[0, 1].set_title("high width, low support", **tkw)
    axes[1, 0].plot(g2)
    axes[1, 0].set_title("low width, high support", **tkw)
    axes[1, 1].plot(g3)
    axes[1, 1].set_title("low width, low support", **tkw)
    fig.subplots_adjust(left=0, right=1, wspace=.01, hspace=.15)
    
    fig.set_size_inches(10, 9)


.. raw:: html

  <img width="500" class="padded" src="https://user-images.githubusercontent.com/16495490/168939821-e7946edc-bf28-4c88-bcc5-583e57f1ba90.png">

  
`pack_coeffs_jtfs()`
--------------------

  
.. raw:: html

  <img height="560" class="padded" src="https://user-images.githubusercontent.com/16495490/168933190-3b3ce10b-3513-4ab1-a113-be1e19ddaee5.png">

  <img height="580" class="padded" src="https://user-images.githubusercontent.com/16495490/168933232-7b8f43bc-de6d-4896-9f27-ecccf88ceb1c.png">
  
  <img height="580" class="padded" src="https://user-images.githubusercontent.com/16495490/168933368-daa65eec-920d-4db5-99a5-60d493c7d113.png">


`_energy_correction()` in JTFS `core`
-------------------------------------

Energy mismatch due to unpad aliasing, demo -- see `discussion <https://github.com/kymatio/kymatio/discussions/753#discussioncomment-947282>`_  # TODO

.. collapse:: code

  .. code-block:: python
  
    import numpy as np
    from numpy.fft import fft, ifft
    from kymatio.visuals import plot, plotscat
    
    def E(x):
        return np.sum(np.abs(x)**2)
    
    np.random.seed(10)
    x = np.random.randn(256)
    xf = fft(x)
    xf[16:-15] = 0  # ensure zero alias
    x = ifft(xf)
    sm = np.abs(x.imag).sum()
    assert sm < 1e-14, sm
    x=x.real
    
    x0 = x.copy()
    x1 = x[::8]
    
    _t = lambda txt: (txt, {'fontsize': 22})
    ckw=dict(w=.7,h=.9)
    plot(x0, title=_t("x0"), show=1,**ckw)
    plotscat(x1, title=_t("x1=x0[::8]"), show=1,**ckw)
    
    e0s, e1s = [E(x0)], [E(x1)*8]
    for i in range(1, len(x1)+1):
        e0s.append(E(x0[:-8*i]))
        e1s.append(E(x1[:-i])*8)
    plotscat(e0s, auto_xlims=0,**ckw)
    plotscat(e1s, show=1, title=_t("E(x0[:-8*i]), E(x1[:-i])*8"),**ckw)


.. raw:: html

  <img width="600" class="padded" src="https://user-images.githubusercontent.com/16495490/168958370-d2530880-e991-434f-a093-ceae2fc26f04.png">
  
  
happens even if we're very safe... (change `16:-15` to `4:-3` in code, can losslessly subsample 4x as much now)


.. raw:: html

  <img width="610" class="padded" src="https://user-images.githubusercontent.com/16495490/168958734-85532dcf-fe47-4563-ae17-e469b60c7814.png">
