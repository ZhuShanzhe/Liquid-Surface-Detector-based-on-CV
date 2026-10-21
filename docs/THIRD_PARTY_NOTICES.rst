第三方组件、许可证与源码获取
==========================

本发行包包含独立 Blender 进程、CPython、PySide6/Qt WebEngine、PyTorch、
torchvision、NumPy、SciPy、scikit-learn、OpenCV 等组件。各组件的版权和
许可证归原作者；本项目的软件著作权声明不覆盖这些第三方组件。

随包位置
--------

* ``_internal/licenses/python``：按发行包名称保存原 wheel 中的许可证、
  NOTICE、COPYING 与 METADATA（含确切版本及上游链接）。为避免传递依赖
  遗漏，清单也保留构建环境组件的 notices；不表示每个组件都加载执行。
* ``_internal/licenses/CPython/LICENSE.txt``：嵌入 CPython 的完整许可。
* ``_internal/licenses/upstream``：LGPL 3、LGPL 2.1、GPL 2、GPL 3 正文，
  以及 Qt 官方 WebEngine 6.140.0 第三方许可页的离线 HTML 快照和来源哈希。
  本 wheel 不提供可访问的 chrome://credits；本快照不冒充二进制自动生成的
  SBOM。Qt 文档的 GFDL 1.3 正文也随附。
* ``_internal/vendor/blender/copyright.txt`` 和 ``license``：Blender 4.2.0
  原发行包的版权、GPL 与第三方库许可；原目录完整保留。

PySide6 wheel 元数据声明 LGPL/GPL 与商业授权选项，但当前 wheel 的
``License-File`` 仅列出商业选项说明。因此本包保留该原始说明，并另附
开源许可证正文；这不代表用户取得 Qt 商业授权。Qt WebEngine 中 Chromium
的第三方条款也必须同时保留，官方许可页已合并为离线 HTML。
依据：`Qt 许可说明 <https://doc.qt.io/qt-6/licensing.html>`_、
`Qt WebEngine 许可说明 <https://doc.qt.io/qt-6/qtwebengine-licensing.html>`_。

本程序以独立 DLL/目录方式加载 Qt，没有将 Qt 静态链接到项目算法。
不得删去第三方通知或限制所选开源许可允许的库替换、调试修改等行为。
再次发布二进制时，发行者仍须依照实际采用的 GPL/LGPL 条款提供对应源码、
必要构建信息或适用的源码提供安排；仅附此说明不替代这些义务。

上游源码入口与核查边界
----------------------

* Blender 4.2.0：``https://projects.blender.org/blender/blender/src/tag/v4.2.0``；
  源码发行目录 ``https://download.blender.org/source/``。
* Qt 6.12.0：``https://code.qt.io/cgit/qt/qtbase.git/``，对应 v6.12.0；
  Qt for Python/PySide ``https://code.qt.io/cgit/pyside/pyside-setup.git/``。
* WebEngine 单独发行版本以随包 ``PySide6_WebEngine`` METADATA 为准
  （本次 6.12.0.140）；源码仓库为
  ``https://code.qt.io/cgit/qt/qtwebengine.git/``。2026-10-21 通过
  ``git ls-remote`` 已核实上述 QtBase/PySide/Blender 的正式标签；但
  WebEngine 仓库仅显示 ``v6.140.0-beta1``、``beta2`` 和 ``rc``，尚未
  查到 ``v6.140.0`` 正式标签。wheel 版本与文档版本不能替代确切源码
  提交核对；不能擅自将 rc 当作对应源码。公开分发前需向 Qt 上游确认
  此二进制的对应源码版本及 Chromium 子模块、必要构建材料。
  本地运行时 API 已核实 ``qWebEngineVersion()=6.140.0``、
  ``qWebEngineChromiumVersion()=140.0.7339.225``、
  ``qWebEngineChromiumSecurityPatchVersion()=152.0.7977.83``；
  ``QLibraryInfo.build()`` 为 Qt 6.12.0 / MSVC 2022 shared release。
  API 与 wheel METADATA 都未给出确切提交，后者也无 ``direct_url.json``。
  文档快照与运行时版本相符，但不因此成为该二进制的精确 SBOM。
* PyTorch、torchvision 及其他 Python 组件：每个组件随包 METADATA 的
  Project-URL 与版本字段；许可和 NOTICE 保留原文，没有用概述替代原文。
* PyInstaller bootloader 的许可与 bootloader exception：随包
  ``licenses/python/pyinstaller`` 中的 ``COPYING.txt``。

当前目录是许可证/来源清单，不是法律意见，也不是商业发行合规认证。
对外提供可执行文件前应核对对应源码获取安排和最终依赖清单；用户无需
联网即可阅读随包许可、运行软件或生成新的仿真帧。
