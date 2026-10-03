# 仓库整理依据与发布边界

整理日期：2026-10-03。本次采用 GitHub 官方文档中 README 的常见内容组织：项目作用、用途、开始使用、帮助与维护信息；详细操作放入独立文档，仓库内文件通过相对链接连接。

## 已实际查阅的官方来源

以下页面在本次整理时通过只读 HTTP 获取原文，并用于核对发布格式和边界：

- [About the repository README file](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- [Ignoring files](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)
- [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [Setting your commit email address](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)
- [Email addresses reference](https://docs.github.com/en/account-and-profile/reference/email-addresses-reference)
- [MIT License / Choose a License](https://choosealicense.com/licenses/mit/)；GitHub 官方模板用于根层 `LICENSE`。

这些是格式和操作依据，不表示 GitHub/OpenAI 审核或认可本项目及其效果，也不存在“套用格式就达到算法质量标准”的含义。

## 纳入的内容

- 自有 Python/前端源码、现有测试和研究/启动脚本。
- 更新后的 README、协作说明、贡献说明、最小 demo requirements。
- 经过整理的运行、项目状态、模型资源和第三方来源说明。

现有代码结构不改动，保留导入、API 与输出契约。`.gitattributes` 保留既有研究源码字节，避免换行自动转换破坏来源哈希；新文档统一 LF。

## 明确排除

私有视频和标注、缓存、训练模型包、checkpoint、归档、图片/视频、日志、临时状态、旧实验报告及未经完整许可审核的第三方源码副本。`.gitignore` 使用根目录和文档白名单；新增发布路径必须显式审核。

不安装新扫描工具，不将代码上传在线扫描服务；使用本地文本/路径/大小检查并检查实际 Git index。初次发布前不存在旧 Git 历史；后续更新仍须检查将推送的可达历史，不能只扫描工作目录。模式扫描只能覆盖所检查的风险类型，不是任何未知秘密或隐私都绝不会出现的保证。

## 发布授权与核验

实际 owner/repository、私有或公开、目标分支、上传范围、作者身份。缺少 Git 提交身份时，仅在用户同意后配置当前仓库，不更改全局设置。仓库创建与精确分支 push 分开，完成后比较本地与远端 SHA 和实际可见性。

项目所有者已明确同意公开发布源码、采用 MIT 许可证，并使用 GitHub 账号署名与隐私提交邮箱（仅配置本仓库）。目标为 `Seele-111/jepa-system`，发布分支为 `main`。MIT 只覆盖项目所有者有权授权的自有代码，不重新授权第三方资源。

本次没有附加 Actions、Pages、Release、协作者邀请或生产部署。公开或私有仓库都不能提交真实密钥或用户数据。
