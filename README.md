# BIA601-HW-Delivery-Routes

## تخطيط جولة توصيل بين المواقع

**BIA601 — Intelligent Algorithms / الخوارزميات الذكية — الوظيفة الثالثة (F25) — مجموعة BIA601-HWخوارزميات ذكية**

المستودع: https://github.com/karam267/BIA601-HW-Delivery-Routes

المشروع يحدد مسار مركبة توصيل تنطلق من مستودع، تزور مجموعة مواقع مرة واحدة، ثم تعود إلى المستودع بأقل مسافة (مسألة البائع المتجول المتماثلة TSP). نقارن خوارزمية جينية (GA) وسرب جسيمات متقطع (PSO) ومستعمرة النمل (ACO)، مع جار أقرب (NN) كمرجع سريع وحل دقيق (MILP مع منع الدورات الجزئية) لإثبات المسافة المثلى. الخوارزميات الذكية الثلاث منفّذة من الصفر في `project/search.py`.

A symmetric travelling-salesman delivery tour: start at depot 0, visit every site once, and return, minimizing rounded Euclidean length. GA, discrete PSO, and ACO are implemented from scratch and compared on small, medium, and large synthetic maps.

## فريق العمل — مجموعة BIA601-HWخوارزميات ذكية

| العضو | GitHub | المسؤولية | القضية |
|---|---|---|---|
| كرم محمد جمال الصعيدي | [karam267](https://github.com/karam267) | صياغة مسألة الجولة وتوليد الخرائط | [#1](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/1) |
| سلاف زين الدين | [solaf-z](https://github.com/solaf-z) | تنفيذ الخوارزمية الجينية GA | [#2](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/2) |
| عامر إدريس | [Amer-idrees](https://github.com/Amer-idrees) | تنفيذ خوارزمية PSO | [#3](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/3) |
| باسل محمد سمير عبد الرحمن | [bassel1233](https://github.com/bassel1233) | تنفيذ خوارزمية ACO | [#4](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/4) |
| عبدالله محمد بركات | [amb888](https://github.com/amb888) | المرجع الدقيق والتحقق من الجولات | [#5](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/5) |
| فيصل هيثم دالي | [faisaldaliite-gif](https://github.com/faisaldaliite-gif) | تنفيذ التجارب وتحليل النتائج | [#6](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/6) |
| بشار ياسر المحرز | [xgsdhdsd](https://github.com/xgsdhdsd) | الجداول والرسوم | [#8](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/8) |
| رؤى القالش | [roaaalkalesh](https://github.com/roaaalkalesh) | كتابة التقرير وتدقيقه | [#9](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/9) |
| مريم (Maryam Promotion) | [maryampromotion2-droid](https://github.com/maryampromotion2-droid) | تنظيم المستودع ودليل التشغيل وواجهة العرض | [#7](https://github.com/karam267/BIA601-HW-Delivery-Routes/issues/7) |

مساهمة كل عضو موثّقة في التقرير (`Report.docx`، جدول 7) وفي هذا الملف.

## التشغيل

Python 3.12 أو أحدث. ثبتت التجربة المرفقة على Python 3.12.14 مع إصدارات المكتبات في `project/requirements.txt`. من مجلد `project`:

```text
python -m pip install -r requirements.txt
python experiment.py
python check_results.py
```

ينشئ الأمر الثاني نتائج التجربة الكاملة داخل `results`. المسارات محسوبة من موقع البرنامج. النتائج الأصلية تبقى محفوظة عند استخدام مجلد آخر لتجربة سريعة:

```text
python experiment.py --repeat 2 --output ../trial_results
python check_results.py --results ../trial_results
```

ولإعادة حساب المراجع الدقيقة وإعادة أول تشغيل لكل خوارزمية وحالة، ومقارنة المسارات وسجل التقارب كاملا:

```text
python check_results.py --rerun
```

التفاصيل التقنية والإعدادات والتجارب موجودة في [`project/README.md`](project/README.md).

## النتائج والعرض

- التقرير: [`Report.docx`](Report.docx)
- لوحة النتائج: افتح [`index.html`](index.html) في المتصفح (ملف مستقل، لا يحتاج تشغيل خادم).
- مخرجات التجارب والرسوم: [`project/results/`](project/results/)

## مخرجات التسليم

المشروع جاهز للتسليم قبل 12 نوفمبر 2026، ويتضمن:

1. الكود المصدري القابل للتشغيل في `project/` (GA و PSO و ACO من الصفر، مع NN وحل مرجعي دقيق للمقارنة).
2. تقريراً مختصراً يشرح المسألة والخوارزميات والتجارب والنتائج ومهمة كل عضو (`Report.docx`).
3. ملف `README.md` هذا مع تعليمات التشغيل وإعادة التجربة عبر `check_results.py`.
