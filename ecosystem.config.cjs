// intent-trade serve starts a background KOL poller when
// config/settings.yaml twitter.auto_poll is true (default).
// Browser "自动刷新" only re-reads SQLite; this process does the pulls.
module.exports = {
  apps: [
    {
      name: 'intent-trade',
      cwd: '/home/IntentTrade',
      script: '/home/IntentTrade/.venv/bin/intent-trade',
      args: 'serve --host 127.0.0.1 --port 8787',
      interpreter: 'none',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      max_restarts: 20,
      min_uptime: '5s',
      error_file: '/home/IntentTrade/logs/pm2-error.log',
      out_file: '/home/IntentTrade/logs/pm2-out.log',
      merge_logs: true,
      time: true,
    },
  ],
};
