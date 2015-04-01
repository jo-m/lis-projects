# Images
Learning and Intelligent Systems - Project 3: <http://las.ethz.ch/courses/lis-s15/>

## Installing Theano

    # if using anaconda
    $HOME/anaconda/bin/pip install theano
    # should print 'theano'
    $HOME/anaconda/bin/python -c 'import theano; print theano.__name__'

    # else
    pip install theano
    # should print 'theano'
    python -c 'import theano; print theano.__name__'

Now, we have to enable GPU support.

* [Linux](http://deeplearning.net/software/theano/install.html#gpu-linux)
* [OS X](http://deeplearning.net/software/theano/install.html#gpu-macos)
* [Windows](http://deeplearning.net/software/theano/install_windows.html#gpu-windows)

After installing, try to run `test_gpu.py`:

    ./run.sh test_gpu.py

If "Used the gpu" is printed at the end, everything works now.

**Warning:** the training and result data must not be checked into this repo (ETH Copyright).
