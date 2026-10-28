# 六阶段独立执行 Prompt

> **当前路线修订**：默认工作流已改为官方 SDXL Base 1.0 + InstantID。下文涉及“完整 PhotoMaker V2 身份模块”的内容均为历史阶段约束，不得用于覆盖当前 InstantID 工作流；新的共同约束是保留 InstantID、antelopev2、ControlNet 和输入/输出单人脸守卫。

以下六个代码块均可单独复制给执行代理，不依赖当前对话记忆。每个代理必须区分本地静态检查与服务器行为测试；所有安装、启动、下载、Conda、模型、API、GPU 和真实浏览器操作只在指定 Ubuntu 22.04 x86_64 服务器执行。报告只能记录实际输出，不得虚构结果。

## 阶段 1：服务器基线与模型验收 Prompt

```text
你是阶段 1 执行代理。先完整阅读仓库根目录 DESIGN.md 和 IMPLEMENTATION_PLAN.md，再阅读 scripts/、config/models.lock.json、config/versions.lock.env、ACCEPTANCE.md；实际脚本/API 是命令依据。目标是在目标服务器建立可复核的安装、双模式、功能、性能和画质基线，而不是宣称预先达标。

全局约束：保持 1024×1024、完整 PhotoMaker V2 身份模块（ID Encoder、InsightFace、UNet LoRA），固定物理 GPU 1 和 CUDA_VISIBLE_DEVICES=1；不得降低 1024 分辨率、不得绕过身份模块。先写测试，再写最小修复；使用测试并保存诚实报告，不得虚构结果，静态 PASS 不算功能通过。30 秒仅为已预热、无排队、单并发 Standard P95 目标。

允许修改：仅在真实失败需要最小修复时修改 src/aigc_photomaker/、scripts/（锁文件除外）、workflows/、web/ 及对应 tests/，并写 reports/ 验收报告。不得修改 config/models.lock.json、config/versions.lock.env、requirements/*.txt、DESIGN.md、IMPLEMENTATION_PLAN.md；不得改 GPU 映射、关闭鉴权或扩大范围。需要模型/依赖锁变更时先停止并请求批准。

服务器命令（按顺序记录退出码，失败则保留输出）：
  unzip aigc-photomaker.zip
  cd aigc-photomaker
  export HF_ENDPOINT=https://hf-mirror.com
  export GITHUB_PROXY=https://ghproxy.net/
  export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  export CUDA_VISIBLE_DEVICES=1
  bash scripts/install.sh
  bash scripts/start.sh
  bash scripts/status.sh
  set -a; source .env; set +a
  python -m pytest -q tests
  bash scripts/smoke_test.sh
  bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000
  bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000
本地只允许：python scripts/static_check.py --all、python -m compileall -q scripts src tests、git diff --check。

输出证据：变更和测试清单；脱敏安装/启动/status 输出；reports/ 下 smoke 与 benchmark 的实际 JSON/Markdown 路径；四风格固定 seed 人工画质表；真实浏览器截图；未解决问题。退出标准：Standard 至少 20 成功、零失败、无 OOM、GPU 1 显存证据且热态 P95 ≤ 30 秒；Turbo 至少 10 成功、零失败、无 OOM；1024 RGB PNG 和完整身份链有证据。未满足就报告失败/阻塞，绝不将目标写成结果。
```

## 阶段 2：Standard 画质调优 Prompt

```text
你是阶段 2 执行代理。先完整阅读 DESIGN.md 和 IMPLEMENTATION_PLAN.md，再读 ROADMAP.md 阶段 2、config/styles.json、src/aigc_photomaker/workflow.py、tests/test_styles.py、tests/test_workflow.py 以及阶段 1 原始报告。只针对已有基线中的 Standard 画质问题做最小、可回退调优。

全局约束：保持 1024×1024、完整 PhotoMaker V2 身份模块和物理 GPU 1（CUDA_VISIBLE_DEVICES=1）；不降低 1024 分辨率、不绕过身份模块。先写测试，再改实现；使用测试并保存诚实报告，不得虚构结果。固定输入/seed 做 A/B，不能把一次好图或静态检查当作质量/性能通过。

允许修改：config/styles.json、src/aigc_photomaker/workflow.py、src/aigc_photomaker/config.py 中既有 Standard 参数、scripts/build_workflows.py、tests/test_styles.py、tests/test_workflow.py、tests/test_config.py，以及重新生成的 workflows/*.json 和 reports/。不得修改 config/models.lock.json、config/versions.lock.env、requirements/*.txt、DESIGN.md、IMPLEMENTATION_PLAN.md；不得改 Turbo 来制造 Standard 对照，不得降低步数到破坏基线或引入新模型。需要新增设置/锁变更先请求批准。

先在服务器运行基线测试，再先添加会失败的画质/工作流回归测试，修改后在服务器执行：
  export CUDA_VISIBLE_DEVICES=1
  python -m pytest -q tests/test_styles.py tests/test_workflow.py tests/test_config.py
  python scripts/build_workflows.py
  bash scripts/stop.sh
  bash scripts/start.sh
  set -a; source .env; set +a
  bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000 --style cyberpunk
对四种风格使用同一批准输入/seed 集分别留存 A/B；其余风格按同一 benchmark 命令改 --style。每次模型/参数变化后重新预热。静态补充：python scripts/static_check.py --all 和 git diff --check。

输出证据：测试先后结果、精确参数 diff、工作流 JSON diff、前后样本索引和至少两名评审的盲评表、每份实际 benchmark 路径及 P50/P95/显存/失败/OOM、采用或拒绝结论。退出标准：预定画质问题改善且关键维度无退化，Standard 至少 20 成功、零失败、无 OOM、P95 ≤ 30 秒，完整身份链和 1024×1024 不变；否则回滚并诚实记为未通过。
```

## 阶段 3：Turbo 时延调优 Prompt

```text
你是阶段 3 执行代理。先完整阅读 DESIGN.md 和 IMPLEMENTATION_PLAN.md，再读 ROADMAP.md 阶段 3、src/aigc_photomaker/workflow.py、src/aigc_photomaker/benchmark.py、tests/test_workflow.py、tests/test_benchmark.py 和已冻结的 Standard/Turbo 基线。目标是实测降低 Turbo 热态时延，同时量化身份和画质代价。

全局约束：保持 1024×1024、完整 PhotoMaker V2 身份模块、物理 GPU 1 和 CUDA_VISIBLE_DEVICES=1；不降低 1024 分辨率、不绕过身份模块。先写测试，再写最小实现；使用测试并保存诚实报告，不得虚构结果。不得删失败样本、以冷/热混合数据或交替模式报告冒充单模式结果。

允许修改：src/aigc_photomaker/workflow.py、src/aigc_photomaker/config.py 中既有 Turbo 参数、src/aigc_photomaker/benchmark.py（仅修正报告缺陷）、scripts/build_workflows.py、tests/test_workflow.py、tests/test_config.py、tests/test_benchmark.py、重新生成的 workflows/turbo_api.json 与 turbo_ui.json、reports/。不得修改 config/models.lock.json、config/versions.lock.env、requirements/*.txt、DESIGN.md、IMPLEMENTATION_PLAN.md；不得改 Standard 基线、GPU 映射、LCM 模型或身份节点。新增依赖/模型先请求批准。

服务器验证（先写失败测试，变更后重新预热）：
  export CUDA_VISIBLE_DEVICES=1
  python -m pytest -q tests/test_workflow.py tests/test_config.py tests/test_benchmark.py
  python scripts/build_workflows.py
  bash scripts/stop.sh && bash scripts/start.sh
  set -a; source .env; set +a
  bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000
  bash scripts/benchmark.sh --alternate-modes --runs 10 --base-seed 13000
  bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000
本地仅补充 python scripts/static_check.py --all、python -m compileall -q scripts src tests、git diff --check。

输出证据：Turbo 参数与图 diff、每组实际报告路径、基线/候选同 seed 的 P50/P95/min/max、GPU 1 峰值、失败与 OOM、Standard/Turbo 配对盲评。退出标准：Turbo 至少 10 成功、零失败、无 OOM，热态时延相对基线有实测改善，身份/画质达到事先确认底线；Standard 门槛无回归。交替模式只作稳定性证据。未达标则恢复 8 步、CFG 1.8、lcm/normal 和原图并报告未通过。
```

## 阶段 4：可靠性/安全加固 Prompt

```text
你是阶段 4 执行代理。先完整阅读 DESIGN.md 和 IMPLEMENTATION_PLAN.md，再读 ROADMAP.md 阶段 4、src/aigc_photomaker/ 中 repository/worker/comfy/task_service/security/images/retention/app、scripts/start.sh、scripts/lib.sh、现有 tests/ 和冒烟输出的恢复说明。只修复有测试或真实故障证据的问题。

全局约束：保持 1024×1024、完整 PhotoMaker V2 身份模块、物理 GPU 1 和 CUDA_VISIBLE_DEVICES=1；不降低 1024 分辨率、不绕过身份模块，不为便利关闭鉴权/校验。先写测试，再写最小实现；使用测试并保存诚实报告，不得虚构结果。安全测试不能泄露真实 Key 或私人面部数据。

允许修改：上述可靠性/安全相关 src/aigc_photomaker/ 文件、scripts/lib.sh、scripts/start.sh、scripts/stop.sh、scripts/status.sh、scripts/smoke_test.sh、对应 tests/、deploy/aigc-photomaker.service（仅实际部署修复）、reports/。不得修改 config/models.lock.json、config/versions.lock.env、requirements/*.txt、DESIGN.md、IMPLEMENTATION_PLAN.md；不得引入 Redis/多 Worker/多用户架构或破坏数据库兼容。需要 schema、依赖或架构决策时先请求批准。

先写会复现问题的测试，然后在服务器按影响范围运行：
  export CUDA_VISIBLE_DEVICES=1
  python -m pytest -q tests
  bash scripts/stop.sh && bash scripts/start.sh
  bash scripts/status.sh
  set -a; source .env; set +a
  bash scripts/smoke_test.sh
  bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000
  bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000
按 smoke 输出分别做 Worker/ComfyUI 终止恢复、取消与失败查询；用无 Key/错 Key、恶意路径/图片、限流、启动 INT/TERM 做负例。保存脱敏日志。静态补充：python scripts/static_check.py --all、git diff --check。

输出证据：失败测试到通过的可追溯记录、恢复时间线和 task_id/prompt_id（无 Key）、重复提交检查、负例响应、浏览器截图、性能回归报告、剩余威胁。退出标准：无重复生成/孤儿任务，取消和失败可查询，秘密与路径防护有效，清理不影响运行任务，Standard/Turbo 门槛无回归；否则回滚最小改动并保留失败证据。
```

## 阶段 5：离线发布 Prompt

```text
你是阶段 5 执行代理。先完整阅读 DESIGN.md 和 IMPLEMENTATION_PLAN.md，再读 DEPLOYMENT.md、ROADMAP.md 阶段 5、scripts/prepare_offline_bundle.sh、scripts/install.sh、scripts/lib.sh、scripts/download_models.sh、THIRD_PARTY_NOTICES.md 和 tests/test_static_check.py。目标是可校验的应用 ZIP + 独立大型离线包在干净断网机器部署，不是只设置离线变量。

全局约束：保持 1024×1024、完整 PhotoMaker V2 身份模块、物理 GPU 1 和 CUDA_VISIBLE_DEVICES=1；不降低 1024 分辨率、不绕过身份模块。先写测试，再写最小实现；使用测试并保存诚实报告，不得虚构结果。静态 hash/关键字检查不等于断网安装成功，本项目不得宣称商业许可已清关。

允许修改：scripts/prepare_offline_bundle.sh、scripts/install.sh、scripts/lib.sh、scripts/download_models.sh 中离线缺陷的最小修复、tests/test_static_check.py 及新增离线静态测试、THIRD_PARTY_NOTICES.md 的可核实来源修正、发布 reports/。不得修改 config/models.lock.json、config/versions.lock.env、requirements/*.txt、DESIGN.md、IMPLEMENTATION_PLAN.md；不得删校验、把大包塞入应用 ZIP、包含 .env/.git/日志/结果或重新下载已有 astronaut 预热素材。许可不确定时链接权威来源并标为待核实，先请求批准再作新声明。

联网构建机（Ubuntu 22.04 x86_64、Python 3.11）：
  export AIGC_OFFLINE=0
  bash scripts/prepare_offline_bundle.sh
应用 ZIP 与 offline/ 分开上传，单层目录；断网目标机先记录管理员网络隔离证据，再执行：
  (cd offline && sha256sum -c SHA256SUMS)
  export AIGC_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
  bash scripts/install.sh
  bash scripts/download_models.sh
  bash scripts/start.sh
  bash scripts/status.sh
  set -a; source .env; set +a
  python -m pytest -q tests
  bash scripts/smoke_test.sh
补做缺文件/篡改负例、断网重启和真实浏览器生成。不得在断网验证阶段运行在线安装或下载。构建前后本地静态命令：python scripts/static_check.py --all、python -m compileall -q scripts src tests、git diff --check。

输出证据：应用/大包文件清单和 hash、SOURCE_COMMITS、SHA256SUMS 原始输出、conda-unpack/no-index 日志、断网方式、测试/冒烟/浏览器报告、许可复核表、秘密扫描。退出标准：所有 hash 匹配、无隐式联网、双模式预热和离线功能通过、负例失败、发布包可重建且无秘密；缺任一项即阻塞发布。
```

## 阶段 6：后续多 GPU/多用户 Prompt

```text
你是阶段 6 设计/执行代理。先完整阅读 DESIGN.md 和 IMPLEMENTATION_PLAN.md，再读 ROADMAP.md 阶段 6、MODEL_GPU_EVALUATION.md、当前 Worker/队列/鉴权实现和真实容量报告。首要产出是新设计，不是直接增加 Worker。

强制决策门：必须先编写多 GPU/多用户新设计并获得需求方明确批准；批准前不得修改任何应用、部署、工作流实现代码。新设计至少定义每卡独立 ComfyUI/Worker 和端口/GPU 绑定、队列公平性、用户身份/配额、数据隔离与保留、故障恢复、数据库迁移、特性开关、回滚和逐卡验收。获批后也只能按新设计允许范围实施，先写测试，再写最小实现。

全局约束：既有单卡路径保持 1024×1024、完整 PhotoMaker V2 身份模块、物理 GPU 1 和 CUDA_VISIBLE_DEVICES=1；不降低 1024 分辨率、不绕过身份模块。使用测试并保存诚实报告，不得虚构结果，不把单并发 30 秒目标外推成多用户吞吐。新卡绑定和新容量门槛必须由批准设计定义。

允许修改（批准前）：仅新建设计提案和容量评估报告，不改现有 DESIGN.md、IMPLEMENTATION_PLAN.md。允许修改（批准后）：严格由批准设计列出的 src/aigc_photomaker/、scripts/、deploy/、tests/ 和迁移文件；不得修改 config/models.lock.json、config/versions.lock.env、requirements/*.txt 或现有规划文档，除非需求方另行明确批准。不得擅自引入支付、多机/Kubernetes、匿名公网服务或降低安全边界。

批准前只做只读容量核查；批准后先写失败测试，再按新设计在服务器运行针对性 python -m pytest -q tests、scripts/status.sh、scripts/smoke_test.sh、scripts/benchmark.sh 及新设计规定的逐卡/多用户安全测试。保留旧路径回归：
  export CUDA_VISIBLE_DEVICES=1
  bash scripts/benchmark.sh --mode standard --runs 20 --base-seed 11000
  bash scripts/benchmark.sh --mode turbo --runs 10 --base-seed 12000
本地仅运行 python scripts/static_check.py --all 和 git diff --check；具体服务器命令参数以实际脚本与批准设计为准，不臆造参数。

输出证据：已批准的新设计（批准记录）、容量瓶颈数据、测试清单、每卡进程/GPU/端口映射、跨用户隔离和故障报告、迁移/回滚演练、旧单卡回归报告。退出标准：全部满足新设计门槛，证明扩容确有益且无跨用户泄露/GPU 争抢，单卡路径可回退。没有批准或证据时只能交付设计/阻塞报告，不得写实现或宣称完成。
```
