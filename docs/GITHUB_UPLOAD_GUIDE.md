# GitHub 업로드 가이드

아래 순서대로 하면 로컬 폴더를 `https://github.com/HAMyookMANG/FAIR` 저장소에 올릴 수 있습니다.

## 1. 저장소 clone

```bash
git clone https://github.com/HAMyookMANG/FAIR.git
cd FAIR
```

## 2. 파일 복사

이 패키지의 파일들을 clone한 `FAIR/` 폴더 안에 복사하세요. 기존 파일이 있으면 필요한 파일만 덮어쓰면 됩니다.

권장 구조:

```text
FAIR/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── notebooks/
├── scripts/
├── src/
└── docs/
```

## 3. Git 상태 확인

```bash
git status
```

## 4. 변경 파일 추가

```bash
git add README.md requirements.txt pyproject.toml .gitignore notebooks scripts src docs
```

## 5. 커밋

```bash
git commit -m "Refactor notebook into Python package"
```

## 6. GitHub에 push

```bash
git push origin main
```

만약 기본 브랜치 이름이 `master`라면 아래를 사용하세요.

```bash
git push origin master
```

## 7. 자주 나는 오류

### Permission denied 또는 authentication failed

GitHub 로그인이 안 된 상태일 수 있습니다. GitHub CLI를 쓰면 편합니다.

```bash
gh auth login
```

### rejected because remote contains work

GitHub에 이미 커밋이 있어서 로컬이 뒤처진 상태입니다.

```bash
git pull origin main --rebase
git push origin main
```

### 데이터셋/모델 pkl이 같이 올라가려는 경우

`.gitignore`에 `outputs/`, `*.pkl`, `*.csv`가 들어가 있으니 보통은 제외됩니다. 그래도 보이면 아래로 제거하세요.

```bash
git rm --cached -r outputs
git rm --cached '*.pkl'
```
