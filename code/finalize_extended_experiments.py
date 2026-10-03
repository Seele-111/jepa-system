#!/usr/bin/env python3
"""Finalize v4-v9 evidence after independent audits; never promotes a model."""
from __future__ import annotations
import importlib, json
from datetime import datetime, timezone
from pathlib import Path
from optimized_grouped_training import ROOT, digest, load_grouped
from run_compact_experiment_v4 import write_new

VERSIONS=('v4','v5','v6r','v7','v8','v9')
MODULES={'v4':'run_compact_experiment_v4','v5':'run_event_experiment_v5','v6r':'run_video_gate_repair_v6r',
         'v7':'run_multiview_experiment_v7','v8':'run_event_compact_experiment_v8','v9':'run_normal_cost_experiment_v9'}
FINAL=ROOT/'output/algorithm-opt-2026-10-02-v9'
DOC=ROOT/'docs/ALGORITHM_OPTIMIZATION_V4_V9_2026-10-02.md'


def read(path):return json.loads(Path(path).read_text('utf-8'))


def main():
    reports={};audits={};assets=[]
    for v in VERSIONS:
        exp=importlib.import_module(MODULES[v]);exp.check_receipt();root=exp.OUT
        reports[v]=read(root/'report.json')
        if reports[v]['status']!='complete':raise ValueError('unfinished experiment')
        ap=root/('independent_gate_audit.json' if v=='v6r' else 'independent_selection_audit.json')
        audits[v]=read(ap)
        if v=='v6r':
            if audits[v]['repair']['status']!='passed' or not audits[v]['before_after_asset_hashes_equal']:raise ValueError('repair audit failed')
        elif audits[v]['status']!='passed':raise ValueError('independent audit failed')
        assets += [root/'protocol.json',root/'training_receipt.json',root/'report.json',ap]
        if (root/'training_report.json').exists():assets.append(root/'training_report.json')
    baseline=reports['v4']['grouped_v1_baseline']['metrics']['all']
    primary={v:reports[v][f'primary_{v}_nested']['metrics']['all'] for v in VERSIONS}
    if any(r['statistical_promotion_passed'] for r in reports.values()):raise ValueError('accepted experiment requires a separate product/promotion decision')
    defaultpath=ROOT/'output/algorithm-opt-2026-10-02-v8/default_compatibility.json'
    default=read(defaultpath)
    if default['status']!='passed' or default['current_runtime_sha256']!=digest(ROOT/'code/optimized_detector.py'):raise ValueError('default compatibility not current')
    for n,h in default['default_model_sha256'].items():
        if digest(ROOT/'models'/n)!=h:raise ValueError('default model changed after check')
        assets.append(ROOT/'models'/n)
    exports={};products={}
    for v in ('v4','v5','v6r','v8','v9'):
        root=ROOT/f'output/algorithm-opt-2026-10-02-{v}'
        products[v]=read(root/'product_validation/report.json');exports[v]=read(root/'deployment_diagnostic/report.json')
        if products[v]['status']!='passed' or not products[v]['default_models_unchanged']:raise ValueError('product parity/default integrity failed')
        if not exports[v]['diagnostic_only'] or exports[v]['default_promoted']:raise ValueError('diagnostic policy violation')
        if digest(root/'deployment_diagnostic/locator_bundle.json')!=exports[v]['bundle_sha256']:raise ValueError('deployment bundle changed')
        assets += [root/'product_validation/report.json',root/'deployment_diagnostic/report.json',root/'deployment_diagnostic/locator_bundle.json']
    originalpath=ROOT/'output/algorithm-opt-2026-10-02-v6/independent_gate_audit.json'
    original=read(originalpath);assets += [originalpath,defaultpath,FINAL/'service_health.json']
    _,records,_=load_grouped()
    dataset={'source':r'C:\Users\admin\Desktop\测试','rows':len(records),'contents':len({r['sha256'] for r in records}),
             'frames':sum(r['frames'] for r in records),'events':sum(r['event_count'] for r in records),
             'normal_rows':sum(not r['labels'].any() for r in records),'duplicate_conflict_preserved':True}
    dataset['normal_rows']=int(dataset['normal_rows']);dataset['abnormal_rows']=dataset['rows']-dataset['normal_rows']
    runtime_sources=['optimized_detector.py','optimized_video_gate_v6.py','optimized_video_gate_v6r.py','optimized_compact_features_v4.py',
        'optimized_compact_model_v4.py','optimized_duration_decoder_v4.py','optimized_event_training_v5.py','optimized_event_compact_v8.py',
        'optimized_normal_cost_v9.py','optimized_feature_view.py','optimized_feature_view_v2.py','optimized_locator.py',
        'optimized_calibration_state.py','optimized_recall_decoder.py','optimized_motion_features.py','optimized_local_motion.py',
        'verify_new_experiment_product.py','verify_extended_default_compatibility.py','export_crossed_experiment_locator.py',
        'export_video_gate_locator_v6r.py','finalize_extended_experiments.py']
    snapshots={}
    for n in runtime_sources:
        dest=FINAL/'final_runtime_sources'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
        snapshots[n]=digest(dest);assets.append(dest)
    checks=[('test_optimized_detector.py','Windows',24,0),('test_new_experiment_runtime.py','Windows',3,1),
        ('test_audit_compact_selection_v4.py','Windows',25,0),('test_optimized_video_gate_v6r.py','Windows',3,1),
        ('test_optimized_video_gate_v6r.py','WSL',4,0),('test_multiview_experiment_v7.py','Windows',6,0),
        ('test_optimized_event_compact_v8.py','WSL',3,0),('test_optimized_normal_cost_v9.py','Windows',3,0),
        ('test_audit_video_gate_v6.py','Windows',10,0),('test_audit_multiview_v7.py','Windows',19,0)]
    validation={'status':'passed','evidence':'Successful tool execution observed in this task; summary, not a standalone rerun log.',
        'tests':[{'suite':s,'platform':p,'passed':n,'skipped':k,'suite_sha256':digest(ROOT/'code'/s)} for s,p,n,k in checks],
        'skip_reason':'Windows has no sklearn; dependent new training/parity cases executed in existing WSL environment',
        'default_bitwise_predictions_checked':154,'portable_fullfit_rows_per_export':77,'fresh_content_checks_per_export':2,
        'rename_checks_per_export':1,'gate_repair_independent_sklearn_contexts':audits['v6r']['repair']['sklearn_probability_comparisons']}
    write_new(FINAL/'focused_validation_summary.json',validation);assets.append(FINAL/'focused_validation_summary.json')
    failures={v:[k for k,value in reports[v]['statistical_promotion_checks'].items() if not value] for v in VERSIONS}
    decision={'status':'complete','date':'2026-10-02','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'role':'repeated_same_development_validation_not_new_blind_test','default_promoted':False,
        'decision':'retain historical v1 standard/fast; no experiment passes all unchanged guards','dataset':dataset,
        'audited_baseline_metrics':baseline,'primary_metrics':primary,'failed_guards':failures,
        'independent_audits':{v:audits[v]['status'] for v in VERSIONS},
        'precision_erratum':{'original_v6_inference_max_difference':original['original_v6_max_abs_difference_inference_only'],
            'original_v6_above_2e_6':original['original_v6_above_2e_6_inference_only'],'original_intervals_changed':original['original_v6_interval_changes'],
            'v6r_independent_sklearn_max_error':audits['v6r']['repair']['helper_vs_sklearn_max_abs_difference'],'original_artifacts_preserved':True},
        'additional_nested_base_fits_this_continuation':60,'nested_base_fits_v4_through_v9_total':290,
        'default_model_sha256':default['default_model_sha256'],'runtime_source_sha256':snapshots,
        'deployment_exports':{v:{'candidate':exports[v]['chosen']['candidate'],'diagnostic_only':True,
            'path':str(ROOT/f'output/algorithm-opt-2026-10-02-{v}/deployment_diagnostic/locator_bundle.json')} for v in exports},
        'notes':['Fixed outer-best never substitutes for nested primary.',
            'v8 final inner aggregate chooses existing compact HGB control; v9 chooses v8 normal2 ET control, not new normal4.',
            'Nested pipeline scores are not unseen accuracy of its single full-fit export.',
            'Bootstrap does not correct for repeated adaptive experimentation.',
            'Sample/rename fresh checks validate functionality, not generalization.']}
    write_new(FINAL/'final_decision.json',decision);assets.append(FINAL/'final_decision.json')
    def row(name,m):
        return f"| {name} | {100*m['iou_0.3']['f1']:.2f}% | {100*m['iou_0.5']['f1']:.2f}% | {100*m['frame']['f1']:.2f}% | {m['normal']['false_positive_videos']}/14 | {m['positive_videos_without_candidate']}/63 |"
    table='\n'.join([row('公平分组 v1（历史算法参照）',baseline)]+[row(v,primary[v]) for v in VERSIONS])
    intervals='\n'.join(f"- {v}：精定位F1差值的描述性95% content-bootstrap区间 [{100*reports[v]['paired_content_bootstrap']['event_f1_05']['percentile_95_interval'][0]:.2f}, {100*reports[v]['paired_content_bootstrap']['event_f1_05']['percentile_95_interval'][1]:.2f}] 个百分点。" for v in VERSIONS)
    links='\n'.join(f"- [{v} 主报告]({(ROOT/f'output/algorithm-opt-2026-10-02-{v}/report.json').as_posix()})；[独立审计]({(ROOT/f'output/algorithm-opt-2026-10-02-{v}'/('independent_gate_audit.json' if v=='v6r' else 'independent_selection_audit.json')).as_posix()})。" for v in VERSIONS)
    exrows='\n'.join(f"| {v} | {exports[v]['chosen']['candidate']} | {products[v]['checks']['0207.mp4']['elapsed_seconds']:.3f}s | {products[v]['checks']['0317.mp4']['elapsed_seconds']:.3f}s | {products[v]['checks']['renamed']['elapsed_seconds']:.3f}s |" for v in exports)
    md=f"""# 算法优化实验 v4–v9：2026-10-02

## 1. 最终决定

**有定位指标提升，但没有任何一版同时通过全部产品护栏。保留现有 v1 standard/fast 默认，不自动升级。**

这是在同一份已反复检查的开发数据上进行的迭代验证，不是新盲测、外部泛化证明或工业可靠性认证。下文F1不是“准确率”。本次续做新增60组base member训练（每组含帧模型与视频模型）；连同v4/v5/v6，共290组分组base member训练。v7与v6r复用封存分数，无新嵌套base训练。

数据来自 `{dataset['source']}`：{dataset['rows']}条标注、{dataset['contents']}个唯一内容SHA、{dataset['frames']}帧、{dataset['events']}个事件、{dataset['normal_rows']}正常/{dataset['abnormal_rows']}异常。两条同内容视频的4帧标注冲突保留；按SHA同组，内层每内容总权重1，不择优删除标注。

## 2. 同内容隔离的正式嵌套结果

| Pipeline | 事件 F1@IoU .3 | 精定位 F1@IoU .5 | Frame F1 | 正常误报 | 异常无任何候选 |
|---|---:|---:|---:|---:|---:|
{table}

这张表评估的是各版同内容隔离的嵌套pipeline，不是某个全量拟合模型在开发集上的成绩；包括v1参照也不能当成当前单个默认模型的未见准确率。

- v4：正常误报降到2，但18条异常没有候选，不能只看误报。
- v5：精定位F1为41.51%、异常空候选降到7，但正常误报升到8。
- v6r：严格深层OOF视频gate仍有7条正常误报。
- v7：固定两视角融合＋内层双错误预算未解决矛盾；4/5折内层没有预算可行配置，显式最小违约回退后精定位指标反而下降，没有改预算包装成功。
- v8：事件均衡训练与掩码紧凑特征真正结合，精定位F1为41.06%，较基线32.65%提高约8.41个百分点，异常空候选15→12。但正常误报5→6，仍不能升级。
- v9：只将正常帧negative质量2→4，空候选降到11；正常误报仍6，精定位弱于v8。未扩代价/阈值网格追分。

“异常无候选”只计完全空输出；错误定位但有区间的视频不算空候选，不可视为完整事件漏检率。

## 3. 不变的部署护栏

以已审计分组v1为基准：F1@IoU .3不下降；F1@IoU .5至少+0.02；Frame F1最多-0.005；正常误报≤5/14；异常空候选≤12/63；portable/fresh误差≤2e-6，另需独立审核与真实输出验证。

v4仅失败空候选护栏；v5/v6r/v8/v9失败正常误报护栏；v7失败精定位与空候选护栏。**没有用fixed outer-best替换正式nested primary，也没有放宽护栏。**

## 4. 实验机制

### v4：掩码紧凑表示＋时长解码

局部运动使用观测支持掩码、3×3 tile压缩统计、FPS相对上下文、有效成对差分、masked robust统计和方向性对比。RGB PCA仅拟合训练内容。ET/HGB固定参数，40个解码配置；精确两状态DP使用按秒归一的logit能量减转移代价，另有hysteresis对照。

### v5：稀疏/短事件均衡

每内容positive质量1，各事件均分再事件内部帧均分；异常背景negative1、正常negative2；别名分质量后class balance与mean1。GT事件时长/位置只用于训练权重，不作为特征。

### v6/v6r：严格深层OOF gate

每outer的inner-fit再deep3，生成gate训练证据；inner-val内容/标签不参与生成gate训练证据的base。Outer gate只使用outer-fit的inner OOF。独立审计核验45个deep分区、90个base fit proof、40个gate状态，v5/v6每个outer/inner分区完全一致，outer gate fit-content恰为outer train且不含outer-val。Gate只改视频证据，不乘帧分数。OOF与full-fit训练规模迁移仍未解决。

### v7：固定融合与内层双预算

v4 compact RGB HGB与v5 event RGB HGB固定.5/.5平均，并有两个单分支对照。内层content-weighted正常FPR≤.30、异常空候选率≤.20；不可行时最小归一化超额回退。解码仍是原40配置，选择锁定后才解码outer。

### v8：交叉训练，不是后混合

将v5事件权重应用到v4紧凑特征，保持v4 ET/HGB参数、PCA、video head。2个recipe×5outer×(3inner＋1outer)=40组新训练；原compact HGB作为同分区/fit-content对照。沿用v6的内层效用，normal/empty系数均.30，但只用原40配置。

### v9：单一正常负样本代价

只训练ET新recipe：正常negative4、事件positive1/异常背景1不变，再class balance；video head仍normal2。5×4=20组新训练，v8 normal2 ET为对照。两类模型完全同分区与seed，没有按outer视频错误添加特征。

没有文件名/哈希/生成器/GT时间位置特征。哈希仅用于分组、来源与开发集提示。上下文含未来帧，是离线定位，不承诺在线实时。

## 5. 审核查出的精度缺陷

原v6 fit：float64 mean/scale标准化→clip→float32；旧推理先转mean/scale为float32。实际385个验证推理上下文中127个偏差>2e-6，最大{original['original_v6_max_abs_difference_inference_only']:.9g}。

保留原v6源码/缓存/报告/模型，新增v6r独立namespace和显式numeric_revision。只修标准化顺序，不重训嵌套base/gate，不改分组、参数、候选、种子或选择规则。原60配置的23100次区间比较无变化，nested primary也未变，但不因此认可旧精度。

v6r以保存参数独立重建sklearn LogisticRegression、不调用fit，1309个实际输入上下文：helper vs sklearn误差0；vs独立float64 reference最大2.22e-16；存盘概率复算误差0。最终全量导出也以独立sklearn检查。**原v6旧导出的自引用parity不是独立正确性证明，后续使用v6r。**

## 6. 功能、速度与默认兼容

以下均为内层选择的全开发拟合诊断模型，不是已升级默认。每版77条portable与拟合reference误差≤2e-6、所有区间一致；0207/0317/0207改名均重新从像素提取所需特征、无结果缓存、H264完整帧数、证据/区间与离线记录一致。

| Export | 全量内层选择结果 | 0207 | 0317 | 0207改名 |
|---|---|---:|---:|---:|
{exrows}

**v8最终选择旧compact HGB对照，v9选择v8 normal2 ET对照，不是新normal4。** 不按outer成绩强换新recipe。单个full-fit导出不能继承nested pipeline的“未见准确率”；这也显示inner平均/最终模型的选择不稳定与迁移。

功能验证只含32帧/61帧的两个开发内容，加改名检查；约4.5–5.6秒为本机已热模型单次耗时，不代表冷启动/P95/吞吐/实时性能或新视频泛化。

当前runtime与v2封存runtime：两个默认各77条、共154条，frame/effective score逐点bitwise相同，video/所有区间一致，模型SHA不变。5002 app与5004 worker健康；worker非busy、RGB/V-JEPA/I-JEPA已热。未重启/替换原演示。

## 7. 不确定性与瓶颈

{intervals}

以上只是描述性同内容bootstrap，未校正多轮自适应实验/选最好，不称为显著性认证。正常仅14条，一个视频即约7.14个百分点；两个漂亮演示不代表可靠正常特异度。

保守表示减少正常误报，事件训练减少空输出，但尚未稳定兼得。阈值/预算、融合、gate或正常权重加倍都没同时突破护栏。改善来自表示/训练目标/定位头的组合，不能全部归功于JEPA；早期运动/RGB也有效与JEPA是否真接入是两回事。当前corrected仍用真实掩码predictor，无proxy fallback，本轮未重训JEPA backbone。

继续宣称泛化改善前需要真正独立的新内容、正常硬负样本与边界审核；继续追这77条不能替代独立验证。

## 8. 入口与复现边界

{links}
- [最终决定]({(FINAL/'final_decision.json').as_posix()})。
- [154条默认回归]({defaultpath.as_posix()})。
- [聚焦验证摘要]({(FINAL/'focused_validation_summary.json').as_posix()})。
- [原v6缺陷审计]({originalpath.as_posix()})。

脚本位于 `{(ROOT/'code').as_posix()}`：run_compact_experiment_v4、run_event_experiment_v5、run_video_gate_experiment_v6、run_video_gate_repair_v6r、run_multiview_experiment_v7、run_event_compact_experiment_v8、run_normal_cost_experiment_v9。Freeze/train/select、导出、独立审计、fresh验证分别执行，输入/源码SHA前后核验。

封存输出exclusive create，同namespace重跑会拒绝覆盖；重训/评选使用新namespace或副本，不删除旧结果。训练与sklearn验证使用已有WSL `/home/zzy/vjepa2-main/vjepa-env/bin/python`，设`CUDA_VISIBLE_DEVICES=`、`OMP_NUM_THREADS=2`、`OPENBLAS_NUM_THREADS=2`；推理/多数测试使用现有Windows Python。无新生产依赖、无大模型下载、无上传、无Git提交、无原视频/标注修改。

SHA证明输入/源码/存盘结果未变，不等同独立重训证明，也不修复早期原始特征的全部历史来源限制。各namespace保留protocol、receipt、source、训练/选择/审计报告；最终runtime快照和SHA清单位于v9，v2/v3历史文档与报告保留。
"""
    with DOC.open('x',encoding='utf-8') as f:f.write(md)
    assets.append(DOC)
    write_new(FINAL/'final_delivery_manifest.json',{'status':'complete','date':'2026-10-02','default_promoted':False,
        'files_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in assets},'runtime_snapshots':snapshots,
        'scope':'Closed development experiments, not blind/generalization/promotion proof'})
    print(json.dumps({'status':'complete','document':str(DOC),'default_promoted':False,'failed_guards':failures},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
