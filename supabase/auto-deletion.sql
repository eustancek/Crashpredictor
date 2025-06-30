-- Function to auto-delete old data
CREATE OR REPLACE FUNCTION auto_delete_old_data()
RETURNS void AS $$
BEGIN
    -- Delete user-specific data older than 30 days
    DELETE FROM multipliers WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM crash_values WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM training_history WHERE timestamp < NOW() - INTERVAL '30 days';
    DELETE FROM accuracy_tracking WHERE timestamp < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;