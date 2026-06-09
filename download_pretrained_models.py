#!/usr/bin/env python3
"""
预训练模型下载脚本 (支持断点续传)

用于提前下载 transformer_embeddings 和 unimol_embedding 所需的预训练模型，
避免在首次使用时等待下载。

用法:
    python download_pretrained_models.py --all                    # 下载所有模型
    python download_pretrained_models.py --transformer all        # 下载所有 transformer 模型
    python download_pretrained_models.py --unimol all             # 下载所有 unimol 模型
    python download_pretrained_models.py --transformer chemberta  # 下载指定 transformer 模型
    python download_pretrained_models.py --unimol v2-84m          # 下载指定 unimol 模型

支持的模型:
    Transformer: chemberta, molt5, chemroberta, biogpt, roberta, matbert, all
    Uni-Mol: v1, v2-84m, v2-164m, v2-310m, v2-570m, v2-1.1B, all
"""

import os
import sys
import json
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
PRETRAINED_MODELS_DIR = PROJECT_ROOT / "pretrained_models"
PRETRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 镜像源列表 (优先使用国内镜像)
HF_MIRRORS = [
    'https://huggingface-mirror.com',   # 镜像源 1 (国内，推荐)
    'https://hf-mirror.com',            # 镜像源 2 (国内)
    'https://huggingface.co',           # 官方源 (国外)
]

# Transformer 模型配置
TRANSFORMER_MODELS = {
    "chemberta": {
        "model_name": "seyonec/ChemBERTa-zinc-base-v1",
        "cache_subdir": "chemberta_models",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModel",
    },
    "molt5": {
        "model_name": "laituan245/molt5-large",
        "cache_subdir": "molt5_models",
        "tokenizer_class": "T5Tokenizer",
        "model_class": "T5EncoderModel",
    },
    "chemroberta": {
        "model_name": "seyonec/PubChem10M_SMILES_BPE_450k",
        "cache_subdir": "chemroberta_models",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModel",
    },
    "biogpt": {
        "model_name": "microsoft/biogpt",
        "cache_subdir": "biogpt_models",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModelForCausalLM",
    },
    "roberta": {
        "model_name": "roberta-base",
        "cache_subdir": "roberta_models",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModel",
    },
    "matbert": {
        "model_name": "M3L/MatBERT",
        "cache_subdir": "matbert_models",
        "tokenizer_class": "AutoTokenizer",
        "model_class": "AutoModel",
    },
}

# Uni-Mol 模型配置
UNIMOL_MODELS = {
    "v1": {"version": "v1"},
    "v2-84m": {"version": "v2", "size": "84m"},
    "v2-164m": {"version": "v2", "size": "164m"},
    "v2-310m": {"version": "v2", "size": "310m"},
    "v2-570m": {"version": "v2", "size": "570m"},
    "v2-1.1b": {"version": "v2", "size": "1.1B"},
}

# 下载状态文件路径
DOWNLOAD_STATUS_FILE = PRETRAINED_MODELS_DIR / ".download_status.json"


def load_download_status() -> Dict:
    """加载下载状态文件"""
    if DOWNLOAD_STATUS_FILE.exists():
        try:
            with open(DOWNLOAD_STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_download_status(status: Dict):
    """保存下载状态文件"""
    try:
        with open(DOWNLOAD_STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print_warning(f"无法保存下载状态: {e}")


def update_download_status(model_type: str, model_key: str, status: str, info: Optional[Dict] = None):
    """更新下载状态"""
    download_status = load_download_status()
    if model_type not in download_status:
        download_status[model_type] = {}

    download_status[model_type][model_key] = {
        "status": status,  # "downloading", "completed", "failed", "partial"
        "last_update": datetime.now().isoformat(),
        "info": info or {}
    }
    save_download_status(download_status)


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message):
    """打印成功信息"""
    print(f"✅ {message}")


def print_error(message):
    """打印错误信息"""
    print(f"❌ {message}")


def print_info(message):
    """打印信息"""
    print(f"ℹ️  {message}")


def print_warning(message):
    """打印警告信息"""
    print(f"⚠️  {message}")


def check_transformer_model_complete(model_path: Path) -> bool:
    """
    检查 Transformer 模型是否完整下载

    检查要点:
    1. 目录是否存在
    2. config.json 是否存在
    3. 模型权重文件是否存在 (pytorch_model.bin 或 model.safetensors)
    4. tokenizer 配置文件是否存在
    """
    if not model_path.exists():
        return False

    # 必要的文件列表
    required_files = ["config.json"]

    # 检查模型权重文件 (可能是 bin 或 safetensors 格式)
    has_model_file = False
    model_files = ["pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json"]
    for mf in model_files:
        if (model_path / mf).exists():
            has_model_file = True
            break

    if not has_model_file:
        return False

    # 检查必要文件
    for rf in required_files:
        if not (model_path / rf).exists():
            return False

    # 检查 tokenizer 相关文件 (至少需要一个)
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "spiece.model"]
    has_tokenizer = False
    for tf in tokenizer_files:
        if (model_path / tf).exists():
            has_tokenizer = True
            break

    return has_tokenizer


def get_transformer_model_size(model_path: Path) -> int:
    """获取模型目录的总大小（字节）"""
    if not model_path.exists():
        return 0

    total_size = 0
    try:
        for item in model_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except Exception:
        pass
    return total_size


def format_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    if size_bytes == 0:
        return "0 B"

    size_units = ["B", "KB", "MB", "GB", "TB"]
    size_idx = 0
    size_val = float(size_bytes)

    while size_val >= 1024 and size_idx < len(size_units) - 1:
        size_val /= 1024
        size_idx += 1

    return f"{size_val:.2f} {size_units[size_idx]}"


def download_file_with_resume(url: str, local_path: Path, headers: Optional[Dict] = None) -> bool:
    """
    带断点续传功能的文件下载

    Args:
        url: 下载链接
        local_path: 本地保存路径
        headers: 请求头

    Returns:
        是否下载成功
    """
    import requests

    headers = headers or {}
    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")

    # 检查临时文件是否存在，获取已下载大小
    downloaded_size = 0
    if temp_path.exists():
        downloaded_size = temp_path.stat().st_size
        headers['Range'] = f'bytes={downloaded_size}-'
        print_info(f"检测到未完成的下载，已下载: {format_size(downloaded_size)}")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        # 如果服务器不支持断点续传，从头开始下载
        if response.status_code == 416:  # Range not satisfiable
            print_info("服务器不支持断点续传，从头开始下载...")
            headers.pop('Range', None)
            downloaded_size = 0
            response = requests.get(url, headers=headers, stream=True, timeout=30)
        elif response.status_code not in [200, 206]:
            print_error(f"下载失败，HTTP状态码: {response.status_code}")
            return False

        # 获取总大小
        total_size = int(response.headers.get('content-length', 0)) + downloaded_size

        # 开始下载
        mode = 'ab' if downloaded_size > 0 and response.status_code == 206 else 'wb'
        with open(temp_path, mode) as f:
            downloaded = downloaded_size
            last_print_time = time.time()

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # 每2秒打印一次进度
                    current_time = time.time()
                    if current_time - last_print_time > 2:
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print_info(f"下载进度: {percent:.1f}% ({format_size(downloaded)} / {format_size(total_size)})")
                        else:
                            print_info(f"已下载: {format_size(downloaded)}")
                        last_print_time = current_time

        # 下载完成，重命名临时文件
        shutil.move(temp_path, local_path)
        return True

    except Exception as e:
        print_error(f"下载出错: {e}")
        return False


def download_transformer_model(
    model_key: str,
    force_redownload: bool = False,
    force_new_download: bool = False
) -> bool:
    """
    下载指定的 Transformer 模型 (transformers 库自动处理断点续传)

    Args:
        model_key: 模型键名 (如 'chemberta', 'molt5' 等)
        force_redownload: 是否强制重新下载（删除已有缓存）
        force_new_download: 是否强制全新下载（使用 force_download=True）

    Returns:
        是否下载成功
    """
    if model_key not in TRANSFORMER_MODELS:
        print_error(f"未知的模型: {model_key}")
        print_info(f"支持的模型: {', '.join(TRANSFORMER_MODELS.keys())}")
        return False

    config = TRANSFORMER_MODELS[model_key]
    model_name = config["model_name"]
    cache_subdir = config["cache_subdir"]
    tokenizer_class_name = config["tokenizer_class"]
    model_class_name = config["model_class"]

    local_model_path = PRETRAINED_MODELS_DIR / cache_subdir / model_name.replace("/", "_")
    temp_model_path = PRETRAINED_MODELS_DIR / cache_subdir / (model_name.replace("/", "_") + ".tmp")

    print_header(f"下载 Transformer 模型: {model_key}")
    print_info(f"模型名称: {model_name}")
    print_info(f"缓存路径: {local_model_path}")

    # 检查状态
    download_status = load_download_status()
    prev_status = download_status.get("transformer", {}).get(model_key, {}).get("status")

    # 检查模型是否已完整存在
    if check_transformer_model_complete(local_model_path) and not force_redownload:
        model_size = get_transformer_model_size(local_model_path)
        print_success(f"模型已完整存在于缓存中 ({format_size(model_size)}): {local_model_path}")
        update_download_status("transformer", model_key, "completed", {"size": model_size})
        return True

    # 检查是否有未完成的下载（临时目录）
    if temp_model_path.exists() and not force_redownload:
        print_info(f"检测到未完成的下载临时目录: {temp_model_path}")
        print_info("transformers 库会自动尝试继续下载...")
    elif temp_model_path.exists() and force_redownload:
        print_info("强制重新下载，删除临时目录...")
        shutil.rmtree(temp_model_path, ignore_errors=True)

    # 如果强制重新下载，删除旧缓存
    if local_model_path.exists() and force_redownload:
        print_info("强制重新下载，删除旧缓存...")
        shutil.rmtree(local_model_path, ignore_errors=True)

    # 更新状态为下载中
    update_download_status("transformer", model_key, "downloading")

    try:
        from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel, AutoModelForCausalLM

        # 获取正确的类
        class_map = {
            "AutoTokenizer": AutoTokenizer,
            "AutoModel": AutoModel,
            "T5Tokenizer": T5Tokenizer,
            "T5EncoderModel": T5EncoderModel,
            "AutoModelForCausalLM": AutoModelForCausalLM,
        }

        tokenizer_class = class_map[tokenizer_class_name]
        model_class = class_map[model_class_name]

        # 准备分词器参数
        tokenizer_args = {'legacy': False} if tokenizer_class_name == 'T5Tokenizer' else {}

        # 尝试从多个镜像源下载
        tokenizer, model = None, None
        last_error = None

        for mirror_idx, mirror_url in enumerate(HF_MIRRORS, 1):
            try:
                print_info(f"尝试从镜像源 {mirror_idx}/{len(HF_MIRRORS)} 下载: {mirror_url}")

                # 设置环境变量
                original_hf_endpoint = os.environ.get('HF_ENDPOINT')
                original_hf_home = os.environ.get('HF_HOME')

                if mirror_url != 'https://huggingface.co':
                    os.environ['HF_ENDPOINT'] = mirror_url

                # 设置临时缓存目录
                temp_cache_dir = temp_model_path
                os.environ['HF_HOME'] = str(temp_cache_dir)

                # 下载分词器
                print_info("正在下载分词器...")
                tokenizer = tokenizer_class.from_pretrained(
                    model_name,
                    **tokenizer_args,
                    force_download=force_new_download
                )

                # 下载模型
                print_info("正在下载模型...")
                model = model_class.from_pretrained(
                    model_name,
                    force_download=force_new_download
                )

                # 恢复环境变量
                if original_hf_endpoint is not None:
                    os.environ['HF_ENDPOINT'] = original_hf_endpoint
                elif 'HF_ENDPOINT' in os.environ:
                    del os.environ['HF_ENDPOINT']

                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                elif 'HF_HOME' in os.environ:
                    del os.environ['HF_HOME']

                print_success(f"从 {mirror_url} 下载成功！")
                break

            except Exception as e:
                last_error = str(e)
                print_error(f"镜像源 {mirror_idx} 失败: {str(e)[:150]}")

                # 恢复环境变量
                if original_hf_endpoint is not None:
                    os.environ['HF_ENDPOINT'] = original_hf_endpoint
                elif 'HF_ENDPOINT' in os.environ:
                    del os.environ['HF_ENDPOINT']

                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                elif 'HF_HOME' in os.environ:
                    del os.environ['HF_HOME']

                if mirror_idx < len(HF_MIRRORS):
                    print_info("尝试下一个镜像源...")
                    time.sleep(2)
                continue

        if tokenizer is None or model is None:
            print_error("所有下载源都失败")
            update_download_status("transformer", model_key, "failed", {"error": last_error})
            return False

        # 保存到本地缓存
        print_info("正在保存模型到缓存...")
        local_model_path.mkdir(parents=True, exist_ok=True)

        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)

        # 清理临时目录
        if temp_model_path.exists():
            shutil.rmtree(temp_model_path, ignore_errors=True)

        # 创建模型信息文件
        model_size = get_transformer_model_size(local_model_path)
        with open(local_model_path / "model_info.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "model_key": model_key,
                "download_time": datetime.now().isoformat(),
                "size": model_size,
                "size_formatted": format_size(model_size),
            }, f, indent=2)

        print_success(f"模型已成功保存到: {local_model_path} ({format_size(model_size)})")
        update_download_status("transformer", model_key, "completed", {"size": model_size})
        return True

    except ImportError:
        print_error("未安装 transformers 库，请运行: pip install transformers")
        update_download_status("transformer", model_key, "failed", {"error": "import_error"})
        return False
    except Exception as e:
        print_error(f"下载失败: {e}")
        update_download_status("transformer", model_key, "failed", {"error": str(e)})
        return False


def check_unimol_model_complete(model_key: str) -> bool:
    """
    检查 Uni-Mol 模型是否已完整下载

    Uni-Mol 模型存储在用户目录的 .cache/unimol_tools/ 下
    """
    try:
        cache_dir = Path.home() / ".cache" / "unimol_tools"
        if not cache_dir.exists():
            return False

        config = UNIMOL_MODELS[model_key]
        version = config["version"]
        size = config.get("size")

        # 根据版本确定检查路径
        if version == "v2":
            model_dir = cache_dir / f"unimolv2_{size}"
        else:
            model_dir = cache_dir / "unimolv1"

        # 检查模型目录是否存在且非空
        if not model_dir.exists():
            return False

        # 检查是否有模型文件（通常是 .pt 或 .pth 文件）
        model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth")) + list(model_dir.glob("*.bin"))
        return len(model_files) > 0

    except Exception:
        return False


def get_unimol_model_size(model_key: str) -> int:
    """获取 Uni-Mol 模型大小"""
    try:
        cache_dir = Path.home() / ".cache" / "unimol_tools"
        config = UNIMOL_MODELS[model_key]
        version = config["version"]
        size = config.get("size")

        if version == "v2":
            model_dir = cache_dir / f"unimolv2_{size}"
        else:
            model_dir = cache_dir / "unimolv1"

        if not model_dir.exists():
            return 0

        total_size = 0
        for item in model_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size
    except Exception:
        return 0


def download_unimol_model(
    model_key: str,
    force_redownload: bool = False,
    resume: bool = True
) -> bool:
    """
    下载指定的 Uni-Mol 模型

    Args:
        model_key: 模型键名 (如 'v1', 'v2-84m' 等)
        force_redownload: 是否强制重新下载
        resume: 是否启用断点续传 (Uni-Mol 由库内部处理，这里只做状态检查)

    Returns:
        是否下载成功
    """
    if model_key not in UNIMOL_MODELS:
        print_error(f"未知的模型: {model_key}")
        print_info(f"支持的模型: {', '.join(UNIMOL_MODELS.keys())}")
        return False

    config = UNIMOL_MODELS[model_key]
    version = config["version"]
    size = config.get("size")

    print_header(f"下载 Uni-Mol 模型: {model_key}")
    print_info(f"版本: {version}")
    if size:
        print_info(f"大小: {size}")

    # 检查是否已存在
    if check_unimol_model_complete(model_key) and not force_redownload:
        model_size = get_unimol_model_size(model_key)
        print_success(f"模型已完整存在于缓存中 ({format_size(model_size)})")
        update_download_status("unimol", model_key, "completed", {"size": model_size})
        return True

    # 检查之前是否下载失败过
    download_status = load_download_status()
    prev_status = download_status.get("unimol", {}).get(model_key, {}).get("status")

    if prev_status == "downloading":
        if resume:
            print_info("检测到上次未完成的下载，将继续下载...")
        else:
            print_info("检测到上次未完成的下载，但禁用断点续传，将重新下载...")
    elif prev_status == "failed" and resume:
        print_info("上次下载失败，将重试...")

    # 如果强制重新下载，删除旧缓存
    if force_redownload:
        print_info("强制重新下载，删除旧缓存...")
        try:
            cache_dir = Path.home() / ".cache" / "unimol_tools"
            if version == "v2":
                model_dir = cache_dir / f"unimolv2_{size}"
            else:
                model_dir = cache_dir / "unimolv1"

            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
        except Exception as e:
            print_warning(f"删除旧缓存失败: {e}")

    # 更新状态为下载中
    update_download_status("unimol", model_key, "downloading")

    try:
        from unimol_tools import UniMolRepr

        # Uni-Mol 模型的下载是自动的，我们只需要初始化模型即可触发下载
        print_info("正在初始化 UniMolRepr (这将触发模型下载)...")

        if version == "v2":
            clf = UniMolRepr(
                data_type='molecule',
                model_name='unimolv2',
                model_size=size,
                remove_hs=False
            )
        else:  # v1
            clf = UniMolRepr(
                data_type='molecule',
                model_name='unimolv1',
                remove_hs=False
            )

        print_success(f"Uni-Mol {model_key} 模型下载完成！")

        # 执行一次简单的预测来确保模型完全加载
        print_info("验证模型...")
        test_smiles = ["CCO"]  # 乙醇
        _ = clf.get_repr(test_smiles, return_atomic_reprs=False)
        print_success("模型验证成功！")

        # 更新状态
        model_size = get_unimol_model_size(model_key)
        update_download_status("unimol", model_key, "completed", {"size": model_size})
        print_info(f"模型大小: {format_size(model_size)}")

        return True

    except ImportError:
        print_error("未安装 unimol_tools 库，请运行: pip install unimol_tools")
        update_download_status("unimol", model_key, "failed", {"error": "import_error"})
        return False
    except Exception as e:
        print_error(f"下载失败: {e}")
        update_download_status("unimol", model_key, "failed", {"error": str(e)})
        return False


def list_available_models():
    """列出所有可用的模型"""
    print_header("可用的预训练模型")

    print("\n📚 Transformer 模型:")
    for key, config in TRANSFORMER_MODELS.items():
        print(f"  • {key:<12} - {config['model_name']}")

    print("\n🧬 Uni-Mol 模型:")
    for key, config in UNIMOL_MODELS.items():
        size_info = f" ({config['size']})" if 'size' in config else ""
        print(f"  • {key:<12} - Uni-Mol {config['version']}{size_info}")

    # 显示下载状态
    download_status = load_download_status()
    if download_status:
        print("\n📊 下载状态:")
        for model_type, models in download_status.items():
            for model_key, info in models.items():
                status_icon = "✅" if info.get("status") == "completed" else "❌"
                size_info = ""
                if info.get("info", {}).get("size"):
                    size_info = f" ({format_size(info['info']['size'])})"
                print(f"  {status_icon} {model_type}/{model_key}{size_info}")


def clean_temp_files():
    """清理所有临时文件"""
    print_header("清理临时文件")

    cleaned = []

    # 清理 Transformer 临时目录
    for cache_subdir in PRETRAINED_MODELS_DIR.iterdir():
        if cache_subdir.is_dir() and cache_subdir.name.endswith(".tmp"):
            try:
                shutil.rmtree(cache_subdir, ignore_errors=True)
                cleaned.append(str(cache_subdir))
            except Exception:
                pass

    # 清理 .tmp 后缀的文件
    for tmp_file in PRETRAINED_MODELS_DIR.rglob("*.tmp"):
        try:
            if tmp_file.is_file():
                tmp_file.unlink()
            elif tmp_file.is_dir():
                shutil.rmtree(tmp_file, ignore_errors=True)
            cleaned.append(str(tmp_file))
        except Exception:
            pass

    if cleaned:
        print_info(f"已清理 {len(cleaned)} 个临时文件/目录")
        for item in cleaned[:5]:  # 最多显示5个
            print(f"  - {item}")
        if len(cleaned) > 5:
            print(f"  ... 还有 {len(cleaned) - 5} 个")
    else:
        print_info("没有需要清理的临时文件")


def main():
    parser = argparse.ArgumentParser(
        description="下载 ChemIA 项目所需的预训练模型 (支持断点续传)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_pretrained_models.py --all
  python download_pretrained_models.py --transformer all
  python download_pretrained_models.py --unimol all
  python download_pretrained_models.py --transformer chemberta --force
  python download_pretrained_models.py --unimol v2-84m
  python download_pretrained_models.py --unimol v1,v2-164m,v2-310m --transformer molt5,chemberta,matbert
  python download_pretrained_models.py --clean  # 清理临时文件
        """
    )

    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--transformer", metavar="MODEL(S)", help="下载指定的 Transformer 模型，多个用逗号分隔 (或使用 'all' 下载全部)")
    parser.add_argument("--unimol", metavar="MODEL(S)", help="下载指定的 Uni-Mol 模型，多个用逗号分隔 (或使用 'all' 下载全部)")
    parser.add_argument("--list", action="store_true", help="列出所有可用的模型")
    parser.add_argument("--force", action="store_true", help="强制重新下载已存在的模型")
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续传，总是从头开始下载")
    parser.add_argument("--mirror", choices=["1", "2", "3"], help="指定镜像源: 1=国内镜像1, 2=国内镜像2, 3=官方源")
    parser.add_argument("--clean", action="store_true", help="清理所有临时文件")
    parser.add_argument("--status", action="store_true", help="显示下载状态")

    args = parser.parse_args()

    # 清理临时文件
    if args.clean:
        clean_temp_files()
        return

    # 显示状态
    if args.status:
        list_available_models()
        return

    # 如果指定了镜像源，调整顺序
    if args.mirror:
        mirror_idx = int(args.mirror) - 1
        if 0 <= mirror_idx < len(HF_MIRRORS):
            HF_MIRRORS.insert(0, HF_MIRRORS.pop(mirror_idx))
            print_info(f"优先使用镜像源: {HF_MIRRORS[0]}")

    # 断点续传模式
    resume = not args.no_resume
    if not resume:
        print_info("断点续传已禁用")

    # 列出可用模型
    if args.list:
        list_available_models()
        return

    # 如果没有参数，显示帮助
    if not (args.all or args.transformer or args.unimol):
        parser.print_help()
        list_available_models()
        return

    results = {"success": [], "failed": []}

    # 下载所有模型
    if args.all:
        print_header("下载所有预训练模型")

        # 下载所有 Transformer 模型
        for key in TRANSFORMER_MODELS:
            if download_transformer_model(key, args.force, args.no_resume):
                results["success"].append(f"transformer:{key}")
            else:
                results["failed"].append(f"transformer:{key}")

        # 下载所有 Uni-Mol 模型
        for key in UNIMOL_MODELS:
            if download_unimol_model(key, args.force, resume):
                results["success"].append(f"unimol:{key}")
            else:
                results["failed"].append(f"unimol:{key}")

    # 下载指定的 Transformer 模型
    if args.transformer:
        if args.transformer.lower() == "all":
            for key in TRANSFORMER_MODELS:
                if download_transformer_model(key, args.force, args.no_resume):
                    results["success"].append(f"transformer:{key}")
                else:
                    results["failed"].append(f"transformer:{key}")
        else:
            # 支持逗号分隔的多个模型
            model_keys = [k.strip().lower() for k in args.transformer.split(",")]
            for key in model_keys:
                if download_transformer_model(key, args.force, args.no_resume):
                    results["success"].append(f"transformer:{key}")
                else:
                    results["failed"].append(f"transformer:{key}")

    # 下载指定的 Uni-Mol 模型
    if args.unimol:
        if args.unimol.lower() == "all":
            for key in UNIMOL_MODELS:
                if download_unimol_model(key, args.force, resume):
                    results["success"].append(f"unimol:{key}")
                else:
                    results["failed"].append(f"unimol:{key}")
        else:
            # 支持逗号分隔的多个模型
            model_keys = [k.strip().lower() for k in args.unimol.split(",")]
            for key in model_keys:
                if download_unimol_model(key, args.force, resume):
                    results["success"].append(f"unimol:{key}")
                else:
                    results["failed"].append(f"unimol:{key}")

    # 打印总结
    print_header("下载结果总结")

    if results["success"]:
        print(f"\n✅ 成功 ({len(results['success'])} 个):")
        for item in results["success"]:
            print(f"  • {item}")

    if results["failed"]:
        print(f"\n❌ 失败 ({len(results['failed'])} 个):")
        for item in results["failed"]:
            print(f"  • {item}")
        print("\n提示: 可以使用 --resume 参数重新运行以继续下载")
        sys.exit(1)
    else:
        print("\n🎉 所有模型下载成功！")


if __name__ == "__main__":
    main()
