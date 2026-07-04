"""邮件通知模块 - 发送训练状态更新"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging
from datetime import datetime
from pathlib import Path


class EmailNotifier:
    """
    邮件通知器

    发送训练关键事件通知（如 New Best 模型）
    """

    def __init__(self,
                 smtp_server: str = "smtp.163.com",
                 smtp_port: int = 465,
                 sender_email: str = "",
                 sender_password: str = "",
                 recipient_email: str = "",
                 enabled: bool = True):
        """
        初始化邮件通知器

        Args:
            smtp_server: SMTP服务器地址
            smtp_port: SMTP端口 (465=SSL, 587=TLS)
            sender_email: 发件人邮箱
            sender_password: 发件人密码/授权码
            recipient_email: 收件人邮箱
            enabled: 是否启用邮件通知
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.enabled = enabled

        self.logger = logging.getLogger('EmailNotifier')

        # 验证配置
        if self.enabled:
            if not sender_email or not sender_password:
                self.logger.warning("邮件通知已启用但缺少发件人配置，将禁用通知")
                self.enabled = False

    def send_new_best_notification(self,
                                     update_count: int,
                                     score: float,
                                     metrics: dict) -> bool:
        """
        发送 New Best 模型通知

        Args:
            update_count: 更新次数
            score: 最新得分
            metrics: 验证指标字典

        Returns:
            bool: 是否发送成功
        """
        if not self.enabled:
            return False

        try:
            # 构建邮件内容
            subject = f"🏆 New Best Model! Score: {score:.4f} (Update #{update_count})"

            body = self._format_metrics_html(update_count, score, metrics)

            # 发送邮件
            success = self._send_email(subject, body, html=True)

            if success:
                self.logger.info(f"✉️ 已发送New Best通知到 {self.recipient_email}")

            return success

        except Exception as e:
            self.logger.error(f"发送邮件失败: {e}")
            return False

    def _format_metrics_html(self, update_count: int, score: float, metrics: dict) -> str:
        """格式化指标为HTML"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checkpoint_path = metrics.get('best_checkpoint', 'checkpoints/best.pt')

        def render_rank_table(title: str, stats: dict, top_k: int = 5, bottom_k: int = 3) -> str:
            if not stats:
                return f"<h3>{title}</h3><p>暂无数据</p>"

            ranked = sorted(stats.items(), key=lambda item: item[1].get('pnl', 0.0), reverse=True)
            top_items = ranked[:top_k]
            bottom_items = ranked[-bottom_k:] if len(ranked) > top_k else []

            html = f"""
            <h3>{title}</h3>
            <table style="border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px;">
                <tr style="background-color: #2196F3; color: white;">
                    <th style="padding: 8px; border: 1px solid #ddd;">分组</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">盈亏</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">交易数</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">胜率</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">平均盈亏</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">多/空</th>
                </tr>
            """
            for idx, (name, item) in enumerate(top_items):
                bg_color = '#c8e6c9' if idx == 0 else '#e8f5e9'
                pnl_color = 'green' if item.get('pnl', 0.0) >= 0 else 'red'
                html += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 6px; border: 1px solid #ddd;"><strong>{name}</strong></td>
                    <td style="padding: 6px; border: 1px solid #ddd; color: {pnl_color};">{item.get('pnl', 0.0):+.4f}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{int(item.get('trades', 0))}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{item.get('win_rate', 0.0):.1f}%</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{item.get('avg_pnl', 0.0):+.5f}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{int(item.get('num_long', 0))}/{int(item.get('num_short', 0))}</td>
                </tr>
                """
            for name, item in bottom_items:
                pnl_color = 'green' if item.get('pnl', 0.0) >= 0 else 'red'
                html += f"""
                <tr style="background-color: #ffebee;">
                    <td style="padding: 6px; border: 1px solid #ddd;">{name}</td>
                    <td style="padding: 6px; border: 1px solid #ddd; color: {pnl_color};">{item.get('pnl', 0.0):+.4f}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{int(item.get('trades', 0))}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{item.get('win_rate', 0.0):.1f}%</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{item.get('avg_pnl', 0.0):+.5f}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">{int(item.get('num_long', 0))}/{int(item.get('num_short', 0))}</td>
                </tr>
                """
            html += "</table>"
            return html

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #2E7D32;">🏆 发现新的最佳模型！</h2>

            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <p><strong>时间:</strong> {timestamp}</p>
                <p><strong>更新次数:</strong> #{update_count}</p>
                <p style="font-size: 18px; color: #1565C0;"><strong>综合得分:</strong> {score:.4f}</p>
                <p><strong>最佳模型:</strong> <code>{checkpoint_path}</code></p>
            </div>

            <h3>📊 核心指标</h3>
            <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
                <tr style="background-color: #e3f2fd;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>总盈亏</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if metrics.get('total_pnl', 0) >= 0 else 'red'};">{metrics.get('total_pnl', 0):+.4f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>交易数</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('num_trades', 0)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>胜率</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('win_rate', 0):.1f}%</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Expectancy</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('expectancy', 0):+.6f}</td>
                </tr>
                <tr style="background-color: #e3f2fd;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Median PnL</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('median_pnl', 0):+.6f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>PnL Std</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('pnl_std', 0):.6f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Profit Factor</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('profit_factor', 0):.2f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Win/Loss</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('win_loss_ratio', 0):.2f}</td>
                </tr>
                <tr style="background-color: #e3f2fd;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Sharpe / Sortino</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('sharpe', 0):.3f} / {metrics.get('sortino', 0):.3f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Trade DD</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('max_drawdown', 0):.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Avg Win / Loss</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('avg_win', 0):+.5f} / {metrics.get('avg_loss', 0):+.5f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Max Win / Loss</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('max_win', 0):+.5f} / {metrics.get('max_loss', 0):+.5f}</td>
                </tr>
            </table>

            <h3>📈 多空与覆盖面</h3>
            <table style="border-collapse: collapse; width: 100%; margin: 10px 0;">
                <tr style="background-color: #e8f5e9;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>多头</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('num_long', 0)}笔 / {metrics.get('long_win_rate', 0):.1f}% / {metrics.get('long_pnl', 0):+.4f}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>空头</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('num_short', 0)}笔 / {metrics.get('short_win_rate', 0):.1f}% / {metrics.get('short_pnl', 0):+.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>活跃标的 / 市场</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('active_symbols', 0)} / {metrics.get('active_markets', 0)}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>盈利标的 / 市场比例</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('profitable_symbols_ratio', 0):.1f}% / {metrics.get('profitable_markets_ratio', 0):.1f}%</td>
                </tr>
                <tr style="background-color: #e8f5e9;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>交易频率</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('trades_per_1k_steps', 0):.1f}/1k步</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>多空平衡</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">ratio={metrics.get('long_short_ratio', 0):.2f}, balance={metrics.get('long_short_balance', 0):.1%}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>平均持仓</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('avg_holding_steps', 0):.1f}步</td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>平均空仓</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{metrics.get('avg_empty_steps', 0):.1f}步</td>
                </tr>
            </table>
        """

        html += render_rank_table("🌍 市场维度表现", metrics.get('market_stats', {}), top_k=6, bottom_k=4)
        html += render_rank_table("🪙 标的维度表现", metrics.get('symbol_stats', {}), top_k=6, bottom_k=4)

        html += f"""
            <div style="margin-top: 20px; padding: 15px; background-color: #fff3e0; border-left: 4px solid #ff9800;">
                <p style="margin: 0;"><strong>💡 说明:</strong> 以上回撤和 Sharpe/Sortino 为本次验证交易序列统计，便于比较不同 best 版本。</p>
            </div>

            <hr style="margin: 20px 0;">
            <p style="color: #666; font-size: 12px;">
                此邮件由 PPO 交易系统自动发送 | {timestamp}
            </p>
        </body>
        </html>
        """

        return html

    def _send_email(self, subject: str, body: str, html: bool = False) -> bool:
        """
        发送邮件

        Args:
            subject: 邮件主题
            body: 邮件正文
            html: 是否为HTML格式

        Returns:
            bool: 是否发送成功
        """
        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            # 添加正文
            if html:
                msg.attach(MIMEText(body, 'html', 'utf-8'))
            else:
                msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 连接SMTP服务器并发送
            if self.smtp_port == 465:
                # SSL
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=10) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
            else:
                # TLS (587端口)
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)

            return True

        except smtplib.SMTPAuthenticationError:
            self.logger.error("SMTP认证失败，请检查邮箱和密码/授权码")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP错误: {e}")
            return False
        except Exception as e:
            self.logger.error(f"发送邮件时出错: {e}")
            return False

    def test_connection(self) -> bool:
        """
        测试邮件配置是否正确

        Returns:
            bool: 测试是否成功
        """
        if not self.enabled:
            self.logger.warning("邮件通知未启用")
            return False

        try:
            self.logger.info("测试邮件连接...")
            subject = "🧪 测试邮件 - PPO交易系统"
            body = f"""
            <html>
            <body>
                <h2>邮件配置测试成功！</h2>
                <p>PPO交易系统邮件通知功能正常工作。</p>
                <p>当训练发现新的最佳模型时，您将收到通知。</p>
                <p style="color: #666; font-size: 12px; margin-top: 20px;">
                    测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </p>
            </body>
            </html>
            """

            success = self._send_email(subject, body, html=True)

            if success:
                self.logger.info(f"✅ 测试邮件已发送到 {self.recipient_email}")
            else:
                self.logger.error("❌ 测试邮件发送失败")

            return success

        except Exception as e:
            self.logger.error(f"测试邮件连接失败: {e}")
            return False


def create_email_notifier(config_path: Optional[str] = None) -> EmailNotifier:
    """
    从配置文件创建邮件通知器

    Args:
        config_path: 配置文件路径 (可选)

    Returns:
        EmailNotifier实例
    """
    # 默认配置
    email_config = {
        'smtp_server': 'smtp.163.com',
        'smtp_port': 465,
        'sender_email': '',
        'sender_password': '',
        'recipient_email': '',
        'enabled': False,  # 默认禁用，需要用户配置后启用
    }

    # 如果提供了配置文件，从中读取
    if config_path and Path(config_path).exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            if 'email' in config:
                email_config.update(config['email'])

    # Filter out non-constructor keys (e.g. _comment)
    valid_keys = {'smtp_server', 'smtp_port', 'sender_email',
                  'sender_password', 'recipient_email', 'enabled'}
    filtered = {k: v for k, v in email_config.items() if k in valid_keys}
    return EmailNotifier(**filtered)
