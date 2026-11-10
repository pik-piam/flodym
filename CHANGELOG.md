# Changelog

## [2.0.0-rc.1](https://github.com/pik-piam/flodym/compare/v1.1.0...v2.0.0-rc.1) (2026-11-10)


### 🔖 Features

* allow missing and extra values per parameter name in parameter readers ([#221](https://github.com/pik-piam/flodym/issues/221)) ([1961d82](https://github.com/pik-piam/flodym/commit/1961d82846cde2a0622d3d658dddf4875b901051))
* attribute access for MFASystem flows, stocks, processes and parameters ([#203](https://github.com/pik-piam/flodym/issues/203)) ([9270773](https://github.com/pik-piam/flodym/commit/92707731db1515fe54f865d9304061f04df4f5e3))
* auto-compute processes ([#202](https://github.com/pik-piam/flodym/issues/202)) ([3e7c89b](https://github.com/pik-piam/flodym/commit/3e7c89bb2b4a2b5d8c3bce35cc2f2d561ef6cb56))
* common pydantic base model forbidding unknown arguments ([#217](https://github.com/pik-piam/flodym/issues/217)) ([5fc5966](https://github.com/pik-piam/flodym/commit/5fc59662575fc7b978b9c3759f7e2ffab64b01e5))
* convert stock and dims inputs, create default cohort dim ([#215](https://github.com/pik-piam/flodym/issues/215)) ([a3ae64d](https://github.com/pik-piam/flodym/commit/a3ae64dd09f78f916a898a54757e77b3489953be))
* dynamic stock compute modes, initial stock and inflow by cohort ([#200](https://github.com/pik-piam/flodym/issues/200)) ([100fdd6](https://github.com/pik-piam/flodym/commit/100fdd6f995c65697967b03d40134552805dee99))
* extensive flag for FlodymArray with adapted dimension handling ([#214](https://github.com/pik-piam/flodym/issues/214)) ([930d2fa](https://github.com/pik-piam/flodym/commit/930d2fa5c3846f94fd6088fccdfe322fda698010))
* is_set property ([#124](https://github.com/pik-piam/flodym/issues/124)) ([0b78f04](https://github.com/pik-piam/flodym/commit/0b78f04b8598be47baf3d697be5ed33b44511b67))
* new lifetime models, competing risks, period effects and mean lifetime ([#201](https://github.com/pik-piam/flodym/issues/201)) ([d991a94](https://github.com/pik-piam/flodym/commit/d991a94c42b5dd24b06659c6e474687b3ec79591))
* new mass balance ([#127](https://github.com/pik-piam/flodym/issues/127)) ([2df41da](https://github.com/pik-piam/flodym/commit/2df41da9e420fecd4d1afc38008a8552e67da382))
* process links ([#126](https://github.com/pik-piam/flodym/issues/126)) ([8df3ff1](https://github.com/pik-piam/flodym/commit/8df3ff1619633631990e8655e584e98c6bf632ea))
* sub-dimensions with aggregation and disaggregation ([#213](https://github.com/pik-piam/flodym/issues/213)) ([43cf24b](https://github.com/pik-piam/flodym/commit/43cf24b663216b20621a81169ba5c1314eb75838))


### 🐛 Bug Fixes

* pin ty to 0.0.100 and fix its new type errors ([#222](https://github.com/pik-piam/flodym/issues/222)) ([8c45f0a](https://github.com/pik-piam/flodym/commit/8c45f0ac9c126fe2fdef3bde46281d1c865d5473))


### 📚 Documentation

* add migration guide from v1 to v2 ([#205](https://github.com/pik-piam/flodym/issues/205)) ([4e3081f](https://github.com/pik-piam/flodym/commit/4e3081f479c523ac9e0cde06ca6c969ff7ebff37))
* use `import flodym as fd` in examples, howtos and docstrings ([#212](https://github.com/pik-piam/flodym/issues/212)) ([a77a082](https://github.com/pik-piam/flodym/commit/a77a082d7b701a5d24b07ad82bb6ca35b05d8fae))


### 🧹 Miscellaneous

* configure prerelease 2.0.0-rc.1 on v2 branch ([7997bcf](https://github.com/pik-piam/flodym/commit/7997bcf39b73d30a92843504d9a2a35f45bf77a9))
* deprecated interfaces to be removed in v3.0.0 ([#206](https://github.com/pik-piam/flodym/issues/206)) ([851fe0c](https://github.com/pik-piam/flodym/commit/851fe0cf406c43f48069fd431142bf790ac4fabd))
* type hints, docstrings, private names and docstring linting ([#210](https://github.com/pik-piam/flodym/issues/210)) ([26a36a5](https://github.com/pik-piam/flodym/commit/26a36a5c63d3cec65b63fce86c21313d7081a413))
* update pre-commit ([#181](https://github.com/pik-piam/flodym/issues/181)) ([ee27639](https://github.com/pik-piam/flodym/commit/ee27639aaaa6355e775b064f13621d6ba1fc1484))
* update pull request template checklist ([#216](https://github.com/pik-piam/flodym/issues/216)) ([1a2fb30](https://github.com/pik-piam/flodym/commit/1a2fb3012ba16d1b98dc35a95040c7b8a3100f46))

## [1.1.0](https://github.com/pik-piam/flodym/compare/v1.0.0...v1.1.0) (2026-10-09)


### 🔖 Features

* add typing to `data_reader` ([#186](https://github.com/pik-piam/flodym/issues/186)) ([ef4eb9e](https://github.com/pik-piam/flodym/commit/ef4eb9e81830d1cc8fa9cdb33c46bfae44043bda))
* enhance typing ([#193](https://github.com/pik-piam/flodym/issues/193)) ([d734e18](https://github.com/pik-piam/flodym/commit/d734e18191714435b53fb6c7cd3341cab630660c))
* improve type hints and accept one dimension in `get_shares_over` ([#196](https://github.com/pik-piam/flodym/issues/196)) ([1750666](https://github.com/pik-piam/flodym/commit/1750666841ce3eb12ae09f1da92c6d957d6e2710))
* lifetime extension by nature and nurture ([#192](https://github.com/pik-piam/flodym/issues/192)) ([faa44ba](https://github.com/pik-piam/flodym/commit/faa44ba26897bf928cb0da11fc054ee1105046a8))
* make dynamic stock model base public ([#190](https://github.com/pik-piam/flodym/issues/190)) ([f754db9](https://github.com/pik-piam/flodym/commit/f754db921fa236720d36872bbd18d205c0d0698b))
* small performance improvements ([#197](https://github.com/pik-piam/flodym/issues/197)) ([e81b91d](https://github.com/pik-piam/flodym/commit/e81b91d68a444e3ccf2f6dd220eff03f67434aea))


### 🐛 Bug Fixes

* keep lifetime parameters intact under lifetime extension by year, and compute it per year ([#198](https://github.com/pik-piam/flodym/issues/198)) ([5321470](https://github.com/pik-piam/flodym/commit/5321470e271c6e0092a0709c4628270ff5d5a7dd))


### 🧹 Miscellaneous

* add agents and claude config ([#188](https://github.com/pik-piam/flodym/issues/188)) ([7e7f94a](https://github.com/pik-piam/flodym/commit/7e7f94ae5eeed20b53565b00726b00ef75ad1f26))
* add type rules and checker step to CI ([#187](https://github.com/pik-piam/flodym/issues/187)) ([e5bb580](https://github.com/pik-piam/flodym/commit/e5bb5804fb72d76ad2613417d1c3605f1e2c3480))
* lifetime model - cosmetic changes ([#199](https://github.com/pik-piam/flodym/issues/199)) ([7f6eb3b](https://github.com/pik-piam/flodym/commit/7f6eb3beeb63f3489504bae83e6b24763c5f2b1b))

## [1.0.0](https://github.com/pik-piam/flodym/compare/v0.8.1...v1.0.0) (2026-08-19)


### 🔖 Features

* add a copy option to fd stocks ([413930f](https://github.com/pik-piam/flodym/commit/413930f7415d2ba8a0056c875d0b1c2bc5137ef5))
* add visualization of the abstract MFA structure ([#180](https://github.com/pik-piam/flodym/issues/180)) ([fa3d336](https://github.com/pik-piam/flodym/commit/fa3d3362e2a009185d99fc3b1efec39105f450e3))
* Complete FlodymArray type hints ([a71edd2](https://github.com/pik-piam/flodym/commit/a71edd2321a1ae0ac0a9731a5dbe42c48ee7681d))
* Convert FlodymArray to subclasses and back ([#169](https://github.com/pik-piam/flodym/issues/169)) ([d11ae27](https://github.com/pik-piam/flodym/commit/d11ae270a67342a4302b727f3ffcedd4d1dfd47d))


### 🐛 Bug Fixes

* add scalar value handling in DataFrameToFlodymDataConverter ([#162](https://github.com/pik-piam/flodym/issues/162)) ([1715113](https://github.com/pik-piam/flodym/commit/1715113e6b2a1d81367db786a28e87a1021185e4))
* enhance `Dimension.from_np` to support pandas string arrays ([#160](https://github.com/pik-piam/flodym/issues/160)) ([351c33d](https://github.com/pik-piam/flodym/commit/351c33d943f7ac3f7ac3295e5deb51334c40a824))
* mark dynamic stock model base class abstract and private ([#183](https://github.com/pik-piam/flodym/issues/183)) ([744b92d](https://github.com/pik-piam/flodym/commit/744b92d18a532cd19eb402860d0fa2f50f66a0c6))


### 📚 Documentation

* add more plots to `example1`, fix typos, add links to API ([#178](https://github.com/pik-piam/flodym/issues/178)) ([c3bb0c9](https://github.com/pik-piam/flodym/commit/c3bb0c9430b0d626bf6386872ae411683510a9f7))
* overhaul README ([d62b236](https://github.com/pik-piam/flodym/commit/d62b236b6b11c87f300e2e57fe2c62ff748bc175))


### 🧹 Miscellaneous

* add uv dependency group for development ([#161](https://github.com/pik-piam/flodym/issues/161)) ([c2f381c](https://github.com/pik-piam/flodym/commit/c2f381ca478926a6a779aedd018e92e09695439b))
* fix (random) test failures due to tkinter ([#177](https://github.com/pik-piam/flodym/issues/177)) ([a72d433](https://github.com/pik-piam/flodym/commit/a72d433a8c0fcf4e73e948471bd0c1062b483a64))
* init release-please version to 0.8.1 ([#171](https://github.com/pik-piam/flodym/issues/171)) ([6f1f5b2](https://github.com/pik-piam/flodym/commit/6f1f5b2d1a6b9b9b0d769024c7e24e5661f4e87a))
* migrate to ruff  ([#167](https://github.com/pik-piam/flodym/issues/167)) ([96188af](https://github.com/pik-piam/flodym/commit/96188af8e0eb8fca33535246471829bfae5191b6))
* only run release PR job in main repo ([#179](https://github.com/pik-piam/flodym/issues/179)) ([792190d](https://github.com/pik-piam/flodym/commit/792190d6eb9adc2787ef452ce2ba6362de4359ce))
* pre-commit autoupdate ([#157](https://github.com/pik-piam/flodym/issues/157)) ([b7365ca](https://github.com/pik-piam/flodym/commit/b7365ca46d02a255f24ef0b1603982c40014ef06))
* remove unnecessary main guard from test files ([b181610](https://github.com/pik-piam/flodym/commit/b18161078cf3aa64e9fce1a22495f06f438531ed))
* target 1.0.0 ([c665d6f](https://github.com/pik-piam/flodym/commit/c665d6f9d08847f29cac8cfb4407b8961c391800))
* target 1.0.0 ([#176](https://github.com/pik-piam/flodym/issues/176)) ([08fcf64](https://github.com/pik-piam/flodym/commit/08fcf640d0bb7259b20b6c289bed25e0cd1fda43))
* update `uv.lock` file after version increment ([#175](https://github.com/pik-piam/flodym/issues/175)) ([832e39d](https://github.com/pik-piam/flodym/commit/832e39d2a33cce030c9e40dad00547f96ee3d4f7))
* update build system to use uv_build ([#163](https://github.com/pik-piam/flodym/issues/163)) ([9032466](https://github.com/pik-piam/flodym/commit/90324662c8ca98ed34ed6991e65cf6a75b1faab9))
* update GitHub Actions workflow to use uv ([#165](https://github.com/pik-piam/flodym/issues/165)) ([ef55ab6](https://github.com/pik-piam/flodym/commit/ef55ab69bd7cfad940f9b094f5518a3656158876))
* update pre-commit ([#173](https://github.com/pik-piam/flodym/issues/173)) ([c9ff4b5](https://github.com/pik-piam/flodym/commit/c9ff4b5c1e5af130bd7afefa595b385d491b6198))
* use conventional commits for pre-commit ([#168](https://github.com/pik-piam/flodym/issues/168)) ([807eee8](https://github.com/pik-piam/flodym/commit/807eee8993d6c8d8b537e0f6b11d5a96c4b84067))
* use release-please to generate release PRs ([#166](https://github.com/pik-piam/flodym/issues/166)) ([d3fec89](https://github.com/pik-piam/flodym/commit/d3fec8910cd93d63d9ec4e595fdd97266e86fcd4))
