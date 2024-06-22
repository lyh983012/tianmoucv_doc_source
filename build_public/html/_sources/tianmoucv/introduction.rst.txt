TianMouCV算法库
===============

这里是代码库的简要介绍和概述。

目前doc基于Tianmoucv-0.3.2版本，已开源
开发版本为0.3.5.4，待开源

README
------------------------------


**如果你用的是preview版本，下面的tianmouc都替换成Tianmoucv_preview**


Unix系统安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




step 1. 安装基本的编译工具

.. code:: bash

    sudo apt-get update
    sudo apt-get install make
    sudo apt-get install cmake
    sudo apt install build-essential
    
注:macOS使用brew安装gcc/g++，install默认使用g++

.. code:: bash

    brew install make
    brew install cmake
    brea install g++,gcc
    
step 2. 安装python和pytorch，尽量使用cuda版本与anaconda

.. code:: bash

    conda create -n tianmoucv --python=3.10
    conda activate tianmoucv
    conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia
  
step 3. 执行自动安装脚本

.. code:: bash

    pip install tianmoucv -i https://pypi.tuna.tsinghua.edu.cn/simple

或者从源码安装(某些特殊情况下.so文件无法自动编译时可使用)

.. code:: bash

    git clone https://github.com/Tianmouc/tianmoucv_preview.git
    cd tianmoucv_preview
    sh install.sh
    


Windows系统安装
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

step 1. 安装基本的编译工具

安装minGW,make,cmake，git 前往对应官网下载

- https://www.mingw-w64.org/

minGW是一个windows上的gnu最小系统，建议直接下载最新版本的预编译压缩包,以支持最新的c++协议

https://github.com/niXman/mingw-builds-binaries/releases

将解压后的文件中的bin文件夹路径添加到系统环境变量中的PATH内，保证命令行可以直接调用g++

（optional）如果你需要使用sdk，那么你还需要进一步安装opencv和cyusb，然后安装cmake

- https://cmake.org/download/
- https://gnuwin32.sourceforge.net/packages/make.htm
- https://git-scm.com/download/win


step 2. 安装python和pytorch，尽量使用cuda版本与anaconda

.. code:: bash

    conda create -n tianmoucv --python=3.10
    conda activate tianmoucv
    conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia
  
step 3. 执行自动安装脚本

.. code:: bash

    git clone git@github.com:Tianmouc/tianmoucv.git
    cd tianmoucv
    ./install.bat

数据解码说明
------------------------------
.. image:: _static/decode.jpg
    :width: 80%
    
解码数据在tianmoucv.data的self.prepocess输入之前完成


.. _index_basic:


说明文档索引
------------------------------


- :ref:`数据 <index_data>`

- :ref:`ISP处理方法 <index_isp>`
    
- :ref:`usb sdk接口 <index_camera>`

- :ref:`算法库 <index_proc>`   

    - :ref:`特征 <index_features>`
    
    - :ref:`追踪 <index_tracking>`

    - :ref:`光流 <index_of>`

    - :ref:`重建 <index_rec>`
    
- :ref:`调用示例 <index_examples>`



.. _index_data:

Tianmocu原始数据读取(data reader)
---------------------------------------------------
  
USB data 读取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: tianmoucv.data.tianmoucData
   :members:
   :no-index:
   :show-inheritance:
   
PICE data 读取
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tianmoucv.data.tianmoucData_pcie
   :members:
   :no-index:
   :show-inheritance:
   
   
.. _index_isp:   

ISP处理方法
---------------------------------------------------

.. automodule:: tianmoucv.isp.isp_basic
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.isp.transform
   :members:
   :no-index:
   :show-inheritance:

   
.. _index_camera:

Tianmocu python相机接口(仅支持usb)
---------------------------------------------------

.. automodule:: tianmoucv.camera.sdk_utils
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.camera.controller
   :members:
   :no-index:
   :show-inheritance:
   

.. _index_proc:   

算法库(proc)
---------------------------------------------------

.. _index_features:   

特征库(features)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tianmoucv.proc.features.diff
   :members:
   :undoc-members:
   :show-inheritance:

.. _index_tracking:   

特征库追踪
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tianmoucv.proc.tracking
   :members:
   :no-index:
   :show-inheritance:
   
   
.. _index_of:   

光流算法库(opticalFlow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tianmoucv.proc.opticalflow.basic
   :members:
   :no-index:
   :show-inheritance:

.. automodule:: tianmoucv.proc.opticalflow.estimator
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.proc.opticalflow.sdraft_net
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.proc.opticalflow.spy_net
   :members:
   :no-index:
   :show-inheritance:

.. _index_rec:

重建算法库(reconstructor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tianmoucv.proc.reconstruct.basic
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.proc.reconstruct.integration
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.proc.reconstruct.tiny_unet
   :members:
   :no-index:
   :show-inheritance:
   
.. automodule:: tianmoucv.proc.reconstruct.fuse_net
   :members:
   :no-index:
   :show-inheritance:

   
.. _index_examples: 

调用示例
-------------------------
.. toctree::
    :maxdepth: 1

    /tianmoucv/feature_matching/feature_matching
    /tianmoucv/feature_tracking_tested/feature_tracking_tested
    /tianmoucv/opticalflow_HS_method/opticalflow_HS_method
    /tianmoucv/opticalflow_LK_method/opticalflow_LK_method
    /tianmoucv/opticalflow_SPYNET/opticalflow_SPYNET
    /tianmoucv/reconstruct_gray/reconstruct_gray
    /tianmoucv/reconstruct_hdr_Laplacian/reconstruct_hdr_Laplacian
    /tianmoucv/reconstruct_tiny_unet/reconstruct_tiny_unet
