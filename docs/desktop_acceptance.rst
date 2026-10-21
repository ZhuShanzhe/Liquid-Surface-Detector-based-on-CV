Windows 离线仿真桌面软件验收边界
================================

本文件记录打包验收要求，不把服务器 GPU 仿真性能直接声明为桌面 CPU
性能，也不把自带仿真当作真实工业相机认证。正式算法以
``create_formal_meter`` 工厂为准，桌面移植只允许资源路径、执行设备和
原生库加载适配，不以旧算法或 sklearn 预测静默替换新正式链路。

资源闭包
--------

* 完整 ``src/liquid_depth`` 运行时、HTML/CSS/JS，以及
  ``configs/release_assets.json`` 和清单中所有 V8 资源。
* ``assets/models/scene_router/scene_head.pt``、``new_view_v14`` 和
  ``small_float_peak`` 目录及哈希清单。joblib 权重要求 scikit-learn 1.9.0。
  ``scene_head.pt`` 的 schema 4 会构造 torchvision MobileNetV3；即使没有
  调用 torchvision 独立推理入口，也不能省略 torchvision。
* Windows 森林预测库三件套：``native_forest.cpp``、
  ``native/forest-win_amd64.dll``、``native/forest-win_amd64.json``。
  DLL 使用 MSVC /MT 静态 CRT，运行时校验 C++ 源码和 DLL 的 SHA-256。
  不需要用户安装编译器；缺失、损坏或架构不符必须明确报错。
* Blender 可执行文件、版本对应的运行库、Python、shader 和授权文件。
  仿真子进程依赖 ``render_scene_console.py``、
  ``generate_synthetic_liquid.py``、``generate_fullframe_v16.py``、
  ``generate_special_scene_strata_v1.py``、``generate_other_scenes_v34.py``、
  ``special_scene_protocol_v1.py``。生成器通过 ``importlib`` 动态读取
  ``src/liquid_depth/simulation.py``，不能只把它编译进主进程后删去源码。
* 标定/相位生成器需要 ``measure_release.py``、``measure_determined.py``、
  ``tof_phase_simulator.py``，及其运行时依赖 NumPy/OpenCV/SciPy。
* QtWebEngine helper、resources、locales、platform plugins 与相关第三方
  notices 必须随发布目录存在；不能依赖开发机 PATH 或网络 CDN。

原生数值等价
------------

``scripts/build_native_forest.py`` 仅用于开发机构建，支持 MSVC x64 或
MinGW-w64。Linux 仍使用已有 GNU 本地构建流程。Windows 内核保持 FP32
输入、FP64 阈值、树顺序 FP64 累加，不使用 fast-math，不改变分支判定。

``tests/test_desktop_native_forest.py`` 验证：

* FP32 相邻数跨越 FP64 阈值时的精确左右分支。
* DLL、源码、清单对应；缺失/损坏不得降级。
* 合成 ExtraTrees 分类/回归结果与 sklearn 对照，绝对容差 1e-14。
* 两组正式森林资产逐模型与 sklearn 对照，绝对容差 1e-14。

2026-10-21：对应正式 ``formal_tof_v14_20261021``、桌面启动器 1.0.0。
Windows Python 3.12.3 / scikit-learn 1.9.0 环境中，原生数值等价 4 项通过，
另加缺失/损坏资源拒绝测试 1 项，共 5 项通过；包括全部正式分类森林和
小漂浮物森林资产，绝对容差 1e-14。

首次 MSVC 14.44 /MT 构建用 ``dumpbin /DEPENDENTS`` 检查仅引用
``KERNEL32.dll``。这不替代整个应用的依赖验收；Torch、Qt、Blender 仍有
各自的原生依赖，须在独立发布目录启动验证。

最终构建使用独立 CPython 3.12.14 venv；避免 Anaconda 的 ICU 动态库
被 Qt 间接载入。独立环境已验证 torch→Qt、Qt→torch 两种导入顺序，
并重跑原生森林、正式 V14、桌面控件与资源测试：48 通过、1 项 Linux
专属检查跳过。不得将开发机 Anaconda DLL 混入发布目录。

离线第三方通知
--------------

``packaging/collect_desktop_licenses.py`` 在构建时收集原 wheel 的许可、
NOTICE、COPYING 和版本 METADATA，以及 CPython 的完整许可。Blender
原目录完整复制并保留其 license 目录。Qt 官方 WebEngine 6.140.0 的
125 个第三方许可页面已保存为离线 HTML（2,287,358 bytes），附每页来源
URL 和 SHA-256；该快照不是二进制生成的 SBOM。构建和应用运行均不需要
重新下载这些许可文件。对应源码提供义务及尚未核实的 WebEngine 正式
源码标签详见 ``THIRD_PARTY_NOTICES.rst``，不能用许可文本收集替代。

真值边界和仿真与实地的区别
------------------------

* 真实液深、液体折射率/散射等仿真控制项只允许进入数据生产者，不能
  直接进入测量逆解。空槽独立量尺、容器尺寸和已知液深先验属于标定输入，
  必须与当前测试液深区分。
* 真实语义和回波来源可用于离线输入资格判断及渲染显示；它们不能作为
  算法液面掩码或漂浮物选择的输入。用户应看到不合格原因，而不是虚构数值。
* 透明/半透明路线需要原始三频四相相关帧、五帧先验、静态无遮挡平面；
  普通 SDK 输出的一张深度图不是等价输入。透明液体与波面/漂浮物的
  组合仍不属于已验证功能，界面必须拒绝而不是强行叠加模块。
* 容器形状/尺寸、相机位置/视场改变后必须重建空槽模型和五帧先验。
  相位生产者的中心 ROI 必须与实际可见容器内液面对应，容器外或遮挡
  像素标记为缺测；这种标记不赋予估计器当前真实高度。
* 理想标定和 ±10 mm 截断高斯标定是两个不同协议，界面和结果必须标注。
  后者为截断前 sigma=10/3 mm 的零均值高斯，不能裁剪测量输出。

交付前验收矩阵
--------------

每项应记录软件版本、资源清单哈希、设备、CPU/GPU、输入控件、是否通过
输入资格检查、测量值、误差、渲染/标定/稳态计算/总耗时及错误原因。

* 从桌面含中文/空格路径启动，不需要 Python、Blender、编译器或安装依赖。
* 无网络时启动、初始化、生成新帧、拖动三维视角和数值显示均正常。
* 圆形、椭圆、矩形容器逐一初始化；改变尺寸/相机距离后重新标定。
* 普通、透明、半透明、低照、眩光、起伏、漂浮物、0%–50% 深度丢失，
  以及已验证的普通液体复合场景；对不可支持组合和不合格输入明确拒绝。
* 同一固定输入在服务器正式链路和桌面设备路径进行液深、最高高度及
  模块诊断对比；报告数值偏差，不能只看应用能打开。
* 视觉只影响最高高度，不回写液深。所有帧结果保留前置资格状态。
* 关闭窗口后回收本次自身后台服务与 Blender 子进程，不影响其他应用。
* 发布目录完整搬移后再启动；应用资源只读，运行数据存入明确的用户目录。

桌面 CPU 耗时不承诺服务器报告中的 0.5 s。渲染时间、初始化时间和
单帧测量时间必须分开展示；只有实测过的设备结果可以写入交付说明。
