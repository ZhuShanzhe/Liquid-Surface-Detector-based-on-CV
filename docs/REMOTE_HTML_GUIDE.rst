服务器支持的 HTML 演示入口
==========================

使用
----

将 ``液深仿真_服务器版.html`` 复制到另一台电脑，双击打开。页面会自动进入
服务器上的完整仿真面板，也可以点击“进入仿真界面”。连接暂时失败时，
本地入口会自动重试，恢复后继续进入。电脑需要联网，无需
安装 Python、Blender、模型、CUDA 或 SSH 客户端。

容器、相机安装、光照、液位、漂浮物、起伏及部分缺失的选项与现有面板
一致。渲染、初始化、场景分类和测量在服务器完成；浏览器绘制三维画面并
显示 RGB、深度、实际高度、估计高度与计算耗时。安装或材质变化仍会自动
重建空槽及相应先验，沿用当前 ToF 链路的场景范围与输入资格检查。

不同浏览器分别保存各自的安装、标定和测量状态。最多保留3个浏览器会话，
服务器串行执行采集和测量；多人同时操作时可能排队。结果不变时只检查
状态版本，避免每秒重复传输整幅图像和点云。

入口文件由运行中的服务器配置生成，含本演示的自动连接凭据；生成后的
私人版本不提交到公开仓库。仓库中的 ``remote_launch.html`` 是无凭据模板。
将生成文件交给需要演示的电脑即可使用。

连接有效期
----------

当前使用 Cloudflare Quick Tunnel 提供临时 HTTPS 地址。服务器和后台服务
需要保持运行；网络断开时浏览器会重试。单独重启测量网关不会改变地址。
重新创建隧道后地址会变化，服务器会自动生成新版 HTML，需要重新复制该
文件到演示电脑。若需要长期固定地址，可另配已有域名和固定隧道。

上游说明：``https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/``。
Quick Tunnel 适用于开发演示，不提供固定地址或在线时长保证。

服务结构
--------

::

  本地 HTML → HTTPS 入口 → 浏览器会话
                          ↓
                    独立标定与运行状态
                          ↓
           同一正式 Console / ToF 测量核心
                          ↓
               服务器 Blender + GPU 推理

公网只连接到 ``remote_console`` 网关；既有控制台继续绑定回环地址。
原有 Host、Origin 和请求令牌检查保持有效。网关认证浏览器后，将请求转发
到该浏览器专用的回环控制台，所有计算输入仍由既有控件校验。

部署文件
--------

* ``src/liquid_depth/remote_console.py``：HTTPS隧道后的本机网关、会话与转发。
* ``scripts/deploy_remote_console.py``：创建独立 supervisor 实例，脱离SSH会话
  运行；不依赖容器内是否具有 systemd。
* ``scripts/run_remote_tunnel.py``：运行隧道并更新服务地址和私人HTML。
* ``scripts/create_remote_launcher.py``：将地址与入口凭据填入HTML模板。
* ``scripts/verify_remote_console.py``：真实浏览器从文件入口进入完整界面，
  验证液位操作、独立浏览器、容器变更、复合和透明类场景。

服务器使用已验证 SHA-256 的 ``cloudflared 2026.10.3``：
``053b80b0a23031a038fdbf3c073f59f82b8f1f152aa67971d7383ffafc0dbab0``。
官方下载：``https://github.com/cloudflare/cloudflared/releases/tag/2026.10.3``。

本次部署目录：``/root/autodl-tmp/liquid-depth-remote-web-20261021``。
入口文件：``private/liquid_remote.html``；访问配置：``private/access.json``；
运行日志：``logs/``；采集及标定数据：``captures/``。

该服务器的 supervisor 是 Go 实现。维护命令为：

::

  /usr/bin/supervisord -c /root/autodl-tmp/liquid-depth-remote-web-20261021/supervisord.conf ctl status
  /usr/bin/supervisord -c /root/autodl-tmp/liquid-depth-remote-web-20261021/supervisord.conf ctl restart liquid-remote-gateway

维护隧道时程序名为 ``liquid-remote-tunnel``；重启它会生成新的临时地址。
服务器整机重启后，需启动这份 ``supervisord.conf`` 并取回新生成的HTML。

验证
----

鉴权、不同浏览器状态隔离、原控制台请求校验、会话数量约束和条件轮询
由 ``tests/test_remote_console.py`` 验证。实际浏览器测试属于部署功能验证，
结果保存为 ``reports/evidence/server-html-v1-20261021/``，不替代正式模型
统计报告。互联网传输和浏览器轮询会增加操作等待时间，面板会分别显示
服务器测量计算、仿真采集和初始化耗时。

2026-10-21：8项网关测试通过；真实浏览器7项工况/操作、12项功能检查通过。
另注入一次入口健康检查失败，自动重试后正常进入并测得液深。公网代理
条件轮询实际返回11次304，避免重复传输未变化的帧。Windows电脑也直接
连接HTTPS地址验证，未建立SSH隧道。
